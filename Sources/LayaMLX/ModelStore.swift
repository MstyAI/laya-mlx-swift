import Foundation
import Hub

public enum LayaModel {
  public static let defaultRepository = "aac6fef/laya-mlx"
  public static let defaultRevision = "20aed815fc6acde75733882e7ec0e3f28aeb9717"

  public static var defaultDirectory: URL {
    FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
      .appending(path: "LayaMLX/model", directoryHint: .isDirectory)
  }

  public static func isPrepared(at directory: URL) -> Bool {
    (try? ModelConfiguration.load(from: directory)) != nil
  }

  @discardableResult
  public static func prepare(
    repository: String = defaultRepository,
    revision: String = defaultRevision,
    at directory: URL = defaultDirectory,
    progress: @escaping @Sendable (Progress) -> Void = { _ in }
  ) async throws -> URL {
    if isPrepared(at: directory) { return directory }
    let hub = HubApi(downloadBase: directory.deletingLastPathComponent())
    let snapshot = try await hub.snapshot(
      from: Hub.Repo(id: repository),
      revision: revision,
      matching: [
        "model.safetensors",
        "rl_agent_config.json",
        "encoder/config.json",
        "tokenizer/*",
      ],
      progressHandler: progress
    )
    if snapshot.standardizedFileURL != directory.standardizedFileURL {
      if FileManager.default.fileExists(atPath: directory.path) {
        try FileManager.default.removeItem(at: directory)
      }
      try FileManager.default.createDirectory(
        at: directory.deletingLastPathComponent(), withIntermediateDirectories: true)
      try FileManager.default.moveItem(at: snapshot, to: directory)
    }
    _ = try ModelConfiguration.load(from: directory)
    return directory
  }
}
