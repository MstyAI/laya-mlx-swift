import Foundation

public enum LayaError: Error, LocalizedError, Sendable {
  case invalidModel(String)
  case invalidQuestion(String)
  case modelNotPrepared(URL)
  case inferenceFailed(String)

  public var errorDescription: String? {
    switch self {
    case .invalidModel(let message), .invalidQuestion(let message), .inferenceFailed(let message):
      message
    case .modelNotPrepared(let directory):
      "Laya is not prepared at \(directory.path). Call LayaModel.prepare first."
    }
  }
}
