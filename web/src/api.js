/** @typedef {{role: "user"|"assistant", content: string, created_at: string}} ChatMessage */

const ERROR_MESSAGES = {
  invalid_request: "Проверьте сообщение: от 1 до 2000 символов, не только пробелы.",
  service_unavailable: "Сейчас не получается обработать запрос. Попробуйте чуть позже.",
  request_timeout: "Ответ занял слишком много времени. Попробуйте чуть позже.",
  internal_error: "Не удалось обработать запрос. Попробуйте чуть позже.",
};

function validDate(value) {
  return typeof value === "string" && Number.isFinite(Date.parse(value));
}

function nonBlank(value) {
  return typeof value === "string" && value.trim().length > 0;
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(path, { ...options, credentials: "same-origin", cache: "no-store" });
  } catch (error) {
    if (options.signal?.aborted) throw error;
    throw new Error("Нет связи с сервером. Проверьте подключение и попробуйте ещё раз.");
  }
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error("Не удалось прочитать ответ сервера. Попробуйте ещё раз.");
  }
  if (!response.ok) {
    // Не выводим произвольный текст тела ошибки или неизвестные технические данные.
    const code = data?.error?.code;
    throw new Error(typeof code === "string" && Object.hasOwn(ERROR_MESSAGES, code)
      ? ERROR_MESSAGES[code] : "Не удалось выполнить запрос. Попробуйте ещё раз.");
  }
  return data;
}

/**
 * Загрузить полную серверную историю с проверкой ролей и содержимого.
 * @param {AbortSignal} [signal] Сигнал отмены при закрытии экрана.
 * @returns {Promise<ChatMessage[]>} Сообщения в серверном порядке.
 */
export async function getHistory(signal) {
  const data = await request("/api/chat/history", { signal });
  if (!Array.isArray(data?.messages) || !data.messages.every((item) =>
    item && ["user", "assistant"].includes(item.role) &&
    nonBlank(item.content) && validDate(item.created_at))) {
    throw new Error("Не удалось загрузить историю. Попробуйте ещё раз.");
  }
  return data.messages.map(({ role, content, created_at }) => ({ role, content, created_at }));
}

/**
 * Передать только исходный текст без автоматического повтора POST.
 * @param {string} message Непустой исходный текст покупателя.
 * @returns {Promise<{answer: string, created_at: string}>} Проверенный ответ API.
 */
export async function sendMessage(message) {
  const data = await request("/api/chat/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  if (!nonBlank(data?.answer) || !validDate(data?.created_at)) {
    throw new Error("Не удалось прочитать ответ помощника. История будет обновлена.");
  }
  return { answer: data.answer, created_at: data.created_at };
}
