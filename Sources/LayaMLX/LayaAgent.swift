import Foundation
import MLX
import MLXNN
import Tokenizers

public actor LayaAgent {
  private let configuration: ModelConfiguration
  private let model: DecisionModel
  private let promptBuilder: PromptBuilder
  private let padTokenID: Int
  private let batchSize: Int
  private var promptCache = PromptPrefixCache()

  public init(modelDirectory: URL, batchSize: Int = 16) async throws {
    guard batchSize > 0 else { throw LayaError.invalidModel("Batch size must be positive.") }
    let configuration = try ModelConfiguration.load(from: modelDirectory)
    guard
      let tokenizer = try await AutoTokenizer.from(
        modelFolder: modelDirectory.appending(path: "tokenizer")
      ) as? PreTrainedTokenizer
    else {
      throw LayaError.invalidModel("The Laya tokenizer could not be loaded.")
    }
    guard let padTokenID = tokenizer.convertTokenToId("[PAD]") else {
      throw LayaError.invalidModel("The tokenizer is missing its padding token.")
    }

    let model = DecisionModel(configuration)
    let weightsURL = modelDirectory.appending(path: "model.safetensors")
    let loaded = try MLX.loadArrays(url: weightsURL)
    let weights = Dictionary(
      uniqueKeysWithValues: loaded.compactMap { key, value in
        key == "temperature" ? nil : (key, value.asType(.float16))
      })
    try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
    model.train(false)
    eval(model.parameters())

    self.configuration = configuration
    self.model = model
    self.promptBuilder = try PromptBuilder(tokenizer: tokenizer, configuration: configuration.agent)
    self.padTokenID = padTokenID
    self.batchSize = batchSize
  }

  public func predict(_ request: LayaRequest) throws -> LayaResult {
    var prepared = [PreparedQuestion]()
    let stateTokens = promptBuilder.tokenizeState(request.state)
    for (id, question) in request.questions {
      let prefix = try promptCache.value(for: question) {
        try promptBuilder.makePrefix(for: question)
      }
      prepared.append(try promptBuilder.prepare(id: id, stateTokens: stateTokens, prefix: prefix))
    }
    prepared.sort { $0.id < $1.id }
    guard !prepared.isEmpty else {
      throw LayaError.invalidQuestion("At least one question is required.")
    }

    var answers = [String: LayaAnswer]()
    for start in stride(from: 0, to: prepared.count, by: batchSize) {
      let questions = Array(prepared[start..<min(start + batchSize, prepared.count)])
      let batch = try InferenceBatch(questions, padTokenID: padTokenID)
      let output = model(
        tokenIDs: batch.tokenIDs,
        attentionMask: batch.attentionMask,
        markerPositions: batch.markerPositions,
        markerMask: batch.markerMask,
        questionTypes: batch.questionTypes
      )
      eval(output.logits, output.actions)
      let rows = output.logits.asArray(Float.self)
      let actionRows = softmaxRows(output.actions)
      let width = output.logits.dim(1)
      for (row, question) in questions.enumerated() {
        let values = Array(rows[(row * width)..<(row * width + question.markers.count)])
        answers[question.id] = makeAnswer(
          question,
          logits: values,
          actProbability: actionRows[row][0]
        )
      }
    }
    return LayaResult(
      answers: answers,
      inputTokens: prepared.reduce(0) { $0 + $1.tokenIDs.count }
    )
  }

  private func makeAnswer(
    _ question: PreparedQuestion,
    logits: [Float],
    actProbability: Float
  ) -> LayaAnswer {
    let temperature = configuration.agent.temperature(for: question.type, optionCount: logits.count)
    let probabilities = softmax(logits.map { $0 / temperature })
    let mapped = Dictionary(
      uniqueKeysWithValues: zip(question.labels, probabilities.map(Double.init)))
    let winningIndex = probabilities.indices.max { probabilities[$0] < probabilities[$1] }!

    switch question.type {
    case .choice:
      return LayaAnswer(
        type: .choice,
        confidence: entropyConfidence(probabilities),
        probabilities: mapped,
        choice: question.labels[winningIndex],
        score: nil,
        booleanProbability: nil,
        actProbability: Double(actProbability)
      )
    case .score:
      let score = zip(probabilities.indices, probabilities).reduce(0) { $0 + Float($1.0) * $1.1 }
      return LayaAnswer(
        type: .score,
        confidence: entropyConfidence(probabilities),
        probabilities: mapped,
        choice: nil,
        score: Double(score),
        booleanProbability: nil,
        actProbability: Double(actProbability)
      )
    case .boolean:
      let yes = probabilities[1]
      return LayaAnswer(
        type: .boolean,
        confidence: Double(max(yes, 1 - yes)),
        probabilities: mapped,
        choice: nil,
        score: nil,
        booleanProbability: Double(yes),
        actProbability: Double(actProbability)
      )
    }
  }
}

extension AgentConfiguration {
  fileprivate func temperature(for type: LayaQuestionType, optionCount: Int) -> Float {
    let size =
      optionCount <= 2 ? "2" : optionCount <= 5 ? "3-5" : optionCount <= 10 ? "6-10" : "11+"
    return temperaturesByOptions["\(type.rawValue):\(size)"] ?? temperatures[type.index]
  }
}

private func softmax(_ values: [Float]) -> [Float] {
  let maximum = values.max()!
  let exponentials = values.map { exp($0 - maximum) }
  let total = exponentials.reduce(0, +)
  return exponentials.map { $0 / total }
}

private func softmaxRows(_ values: MLXArray) -> [[Float]] {
  let width = values.dim(1)
  let flat = values.asArray(Float.self)
  return stride(from: 0, to: flat.count, by: width).map {
    softmax(Array(flat[$0..<$0 + width]))
  }
}

private func entropyConfidence(_ probabilities: [Float]) -> Double {
  guard probabilities.count > 1 else { return 1 }
  let entropy = -probabilities.reduce(0) { $0 + $1 * log(max($1, 1e-12)) }
  return Double(max(0, min(1, 1 - entropy / log(Float(probabilities.count)))))
}
