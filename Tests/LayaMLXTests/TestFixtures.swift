import Foundation
import LayaMLX
import XCTest

func makePublishedAgent() async throws -> LayaAgent {
  let huggingFaceCache =
    NSHomeDirectory()
    + "/.cache/huggingface/hub/models--aac6fef--laya-mlx/snapshots/20aed815fc6acde75733882e7ec0e3f28aeb9717"
  let candidates = [
    ProcessInfo.processInfo.environment["LAYA_MODEL_PATH"],
    LayaModel.defaultDirectory.path,
    huggingFaceCache,
  ].compactMap { $0 }
  guard let path = candidates.first(where: FileManager.default.fileExists(atPath:)) else {
    throw XCTSkip("Prepare the model to run checkpoint tests.")
  }
  return try await LayaAgent(modelDirectory: URL(fileURLWithPath: path))
}

func makeParityRequest() -> LayaRequest {
  LayaRequest(
    state: "The payment was declined twice and the customer is angry.",
    questions: [
      "route": .choice(
        instructions: "Choose the best destination.",
        options: [
          LayaOption("billing", description: "Payment and invoice problems"),
          LayaOption("technical", description: "Product bugs and outages"),
          LayaOption("sales", description: "Purchasing questions"),
        ]
      ),
      "urgent": .boolean(instructions: "This needs urgent human attention."),
      "quality": .score(
        instructions: "Rate the customer sentiment from calm to very upset.",
        levels: ["calm", "concerned", "upset", "very upset"]
      ),
    ]
  )
}

func makeSupportRequest(state: String) -> LayaRequest {
  LayaRequest(
    state: state,
    questions: [
      "department": .choice(
        instructions: "Which department should handle this customer request?",
        options: [
          LayaOption("account", description: "Sign-in, identity, access, and account settings"),
          LayaOption("billing", description: "Charges, invoices, payments, and refunds"),
          LayaOption("sales", description: "Pricing, plans, purchasing, and product evaluation"),
          LayaOption(
            "shipping", description: "Delivery, tracking, damaged packages, and returns in transit"),
          LayaOption("technical", description: "Bugs, outages, errors, and product malfunctions"),
        ]
      ),
      "refund": .boolean(
        instructions: "Does the customer explicitly request a refund or their money back?"
      ),
      "urgency": .score(
        instructions: "How urgent is the request?",
        levels: [
          "routine; no time pressure",
          "soon; should be handled within a few days",
          "urgent; blocking work or needs action today",
          "critical; severe security, safety, or widespread outage",
        ]
      ),
    ]
  )
}
