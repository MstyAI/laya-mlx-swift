import MLX

struct InferenceBatch {
  let tokenIDs: MLXArray
  let attentionMask: MLXArray
  let markerPositions: MLXArray
  let markerMask: MLXArray
  let questionTypes: MLXArray

  init(_ questions: [PreparedQuestion], padTokenID: Int) throws {
    guard !questions.isEmpty else {
      throw LayaError.invalidQuestion("At least one question is required.")
    }
    let length = questions.map(\.tokenIDs.count).max()!
    let markerCount = max(2, questions.map(\.markers.count).max()!)
    var tokens = [Int32](repeating: Int32(padTokenID), count: questions.count * length)
    var attention = [Bool](repeating: false, count: tokens.count)
    var markers = [Int32](repeating: 0, count: questions.count * markerCount)
    var markerValidity = [Bool](repeating: false, count: markers.count)

    for (row, question) in questions.enumerated() {
      for (column, value) in question.tokenIDs.enumerated() {
        tokens[row * length + column] = Int32(value)
        attention[row * length + column] = true
      }
      for (column, value) in question.markers.enumerated() {
        markers[row * markerCount + column] = Int32(value)
        markerValidity[row * markerCount + column] = true
      }
    }

    tokenIDs = MLXArray(tokens, [questions.count, length])
    attentionMask = MLXArray(attention, [questions.count, length])
    markerPositions = MLXArray(markers, [questions.count, markerCount])
    markerMask = MLXArray(markerValidity, [questions.count, markerCount])
    questionTypes = MLXArray(questions.map { Int32($0.type.index) })
  }
}

extension LayaQuestionType {
  var index: Int {
    switch self {
    case .choice: 0
    case .score: 1
    case .boolean: 2
    }
  }
}
