struct PromptPrefixCache {
  private let capacity: Int
  private var values = [LayaQuestion: PreparedPromptPrefix]()
  private var order = [LayaQuestion]()

  init(capacity: Int = 128) {
    self.capacity = capacity
  }

  mutating func value(
    for question: LayaQuestion,
    create: () throws -> PreparedPromptPrefix
  ) rethrows -> PreparedPromptPrefix {
    if let value = values[question] {
      order.removeAll { $0 == question }
      order.append(question)
      return value
    }

    let value = try create()
    if values.count == capacity, let oldest = order.first {
      values.removeValue(forKey: oldest)
      order.removeFirst()
    }
    values[question] = value
    order.append(question)
    return value
  }
}
