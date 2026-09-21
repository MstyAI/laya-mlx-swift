import Foundation
import Tokenizers

struct PromptBuilder {
  let tokenizer: any Tokenizer
  let maxLength: Int
  let headMaxLength: Int
  let classTokenID: Int
  let separatorTokenID: Int
  let maskTokenID: Int
  let maskToken: String

  init(tokenizer: any Tokenizer, configuration: AgentConfiguration) throws {
    self.tokenizer = tokenizer
    maxLength = configuration.maxLength
    headMaxLength = configuration.headMaxLength
    guard let classTokenID = tokenizer.convertTokenToId("[CLS]"),
      let separatorTokenID = tokenizer.convertTokenToId("[SEP]"),
      let maskTokenID = tokenizer.convertTokenToId("[MASK]")
    else { throw LayaError.invalidModel("The tokenizer is missing required ModernBERT tokens.") }
    self.classTokenID = classTokenID
    self.separatorTokenID = separatorTokenID
    self.maskTokenID = maskTokenID
    maskToken = "[MASK]"
  }

  func makePrefix(for question: LayaQuestion) throws -> PreparedPromptPrefix {
    let definition = try QuestionDefinition(question)
    let (tokens, markers) = tokenizePrefix(definition)
    return PreparedPromptPrefix(
      type: definition.type,
      tokens: tokens,
      markers: markers,
      labels: definition.labels,
      scoreLevels: definition.scoreLevels,
      optionCount: definition.options.count
    )
  }

  func tokenizeState(_ state: String) -> [Int] {
    let cleanState = state.replacingOccurrences(of: maskToken, with: " ")
    return tokenizer.encode(text: cleanState, addSpecialTokens: false)
  }

  func prepare(id: String, stateTokens: [Int], prefix: PreparedPromptPrefix) throws
    -> PreparedQuestion
  {
    var tokens = prefix.tokens
    let room = max(0, maxLength - tokens.count - 1)
    tokens += stateTokens.prefix(room)
    tokens.append(separatorTokenID)
    guard prefix.markers.count == prefix.optionCount else {
      throw LayaError.invalidQuestion("Question '\(id)' has too many options for the token budget.")
    }
    return PreparedQuestion(
      id: id,
      type: prefix.type,
      tokenIDs: Array(tokens.prefix(maxLength)),
      markers: prefix.markers.filter { $0 < maxLength },
      labels: prefix.labels,
      scoreLevels: prefix.scoreLevels
    )
  }

  private func tokenizePrefix(_ question: QuestionDefinition) -> ([Int], [Int]) {
    let cleanInstructions = question.instructions.replacingOccurrences(of: maskToken, with: " ")
    var heading = tokenizer.encode(
      text: "\(question.type.rawValue) question: \(cleanInstructions)",
      addSpecialTokens: false
    )
    var options = question.options.map { option in
      [maskTokenID]
        + tokenizer.encode(
          text: " " + option.replacingOccurrences(of: maskToken, with: " "),
          addSpecialTokens: false
        ).prefix(48)
    }
    var headingBudget = headMaxLength - options.reduce(0) { $0 + $1.count }
    if headingBudget < 16 {
      let perOption = max(4, (headMaxLength - 16) / max(1, options.count))
      options = options.map { Array($0.prefix(perOption)) }
      headingBudget = headMaxLength - options.reduce(0) { $0 + $1.count }
    }
    heading = Array(heading.prefix(max(8, headingBudget)))

    var tokens = [classTokenID] + heading + [separatorTokenID]
    var markers = [Int]()
    for option in options {
      markers.append(tokens.count)
      tokens += option
    }
    tokens.append(separatorTokenID)
    return (tokens, markers)
  }
}

struct PreparedPromptPrefix: Sendable {
  let type: LayaQuestionType
  let tokens: [Int]
  let markers: [Int]
  let labels: [String]
  let scoreLevels: [String]
  let optionCount: Int
}

private struct QuestionDefinition {
  let type: LayaQuestionType
  let instructions: String
  let options: [String]
  let labels: [String]
  let scoreLevels: [String]

  init(_ question: LayaQuestion) throws {
    switch question {
    case .choice(let instructions, let options):
      guard !options.isEmpty else {
        throw LayaError.invalidQuestion("Choice options cannot be empty.")
      }
      let labels = options.map(\.label)
      guard Set(labels).count == labels.count else {
        throw LayaError.invalidQuestion("Choice option labels must be unique.")
      }
      self.type = .choice
      self.instructions = instructions
      self.labels = labels
      self.options = options.map {
        guard let description = $0.description, !description.isEmpty else { return $0.label }
        return "\($0.label): \(description)"
      }
      scoreLevels = []
    case .score(let instructions, let levels):
      guard !levels.isEmpty else {
        throw LayaError.invalidQuestion("Score levels cannot be empty.")
      }
      self.type = .score
      self.instructions = instructions
      self.options = levels.enumerated().map { "level \($0.offset): \($0.element)" }
      self.labels = levels.indices.map(String.init)
      scoreLevels = levels
    case .boolean(let instructions, let falseDescription, let trueDescription):
      self.type = .boolean
      self.instructions = instructions
      self.options = [
        "false: \(falseDescription ?? "no, the statement does not hold")",
        "true: \(trueDescription ?? "yes, the statement holds")",
      ]
      labels = ["false", "true"]
      scoreLevels = []
    }
  }
}
