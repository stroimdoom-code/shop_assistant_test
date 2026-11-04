import React, { useEffect, useRef, useState } from "react";
import { getHistory, sendMessage } from "./api.js";

/**
 * Показать единственный экран чата с серверной историей и обработкой ошибок.
 * @returns {React.JSX.Element} Адаптивный интерфейс, доступный с клавиатуры.
 */
export default function App() {
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(true);
  const [historyReady, setHistoryReady] = useState(false);
  const [sending, setSending] = useState(false);
  const [pending, setPending] = useState(null);
  const [historyError, setHistoryError] = useState("");
  const [sendError, setSendError] = useState("");
  const busy = useRef(false);
  const end = useRef(null);
  const input = useRef(null);
  const length = Array.from(draft).length;
  const canSend = historyReady && !loading && !sending && draft.trim().length > 0 && length <= 2000;

  async function refreshHistory(signal) {
    setLoading(true);
    setHistoryError("");
    try {
      setMessages(await getHistory(signal));
      setHistoryReady(true);
    } catch (error) {
      if (signal?.aborted) return;
      setHistoryReady(false);
      setHistoryError(error.message);
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }

  useEffect(() => {
    const controller = new AbortController();
    refreshHistory(controller.signal);
    return () => controller.abort();
  }, []);

  useEffect(() => {
    end.current?.scrollIntoView({ block: "end" });
  }, [messages, pending, historyError, sendError]);

  useEffect(() => {
    if (!sending && historyReady) input.current?.focus();
  }, [sending, historyReady]);

  async function handleSubmit(event) {
    event.preventDefault();
    // Ref закрывает окно до перерисовки React при быстром Enter или двойном клике.
    if (busy.current || !canSend) return;
    busy.current = true;
    const original = draft;
    setSending(true);
    setSendError("");
    setDraft("");
    setPending(original);
    try {
      const result = await sendMessage(original);
      setMessages((previous) => [
        ...previous,
        { role: "user", content: original },
        { role: "assistant", content: result.answer, created_at: result.created_at },
      ]);
      // Синхронизируем также после успеха: cookie могла смениться из-за истечения TTL.
      await refreshHistory();
    } catch (error) {
      setSendError(error.message);
      setDraft(original);
      // Сервер мог сохранить вопрос до сбоя. Заменяем историю, не добавляя дубль.
      await refreshHistory();
    } finally {
      setPending(null);
      setSending(false);
      busy.current = false;
    }
  }

  return (
    <main className="chat-shell" aria-labelledby="chat-title">
      <header className="chat-header">
        <div className="assistant-mark" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
            <path d="M20 11.5a8 8 0 0 1-8 8H5l-3 2v-10a9 9 0 0 1 18 0Z" />
            <path d="M7 11h.01M12 11h.01M17 11h.01" strokeWidth="3" strokeLinecap="round" />
          </svg>
        </div>
        <div>
          <h1 id="chat-title">Помощник магазина</h1>
          <p>О товаре, доставке и возврате</p>
        </div>
      </header>

      <section className="conversation" aria-label="Переписка" aria-busy={loading}>
        <div className="message-list" role="log" aria-live="polite" aria-relevant="additions text">
          {loading && messages.length === 0 && <p className="status">Загружаем историю…</p>}
          {!loading && historyReady && messages.length === 0 && !pending && (
            <div className="welcome">
              <span className="welcome-label">Будем на связи</span>
              <h2>Здравствуйте! Чем могу помочь?</h2>
              <p>Напишите свой вопрос в поле ниже.</p>
            </div>
          )}
          {messages.map((message, index) => (
            <article className={`message message-${message.role}`} key={index}>
              <span className="message-author">{message.role === "user" ? "Вы" : "Помощник"}</span>
              <p>{message.content}</p>
            </article>
          ))}
          {pending && (
            <>
              <article className="message message-user">
                <span className="message-author">Вы</span>
                <p>{pending}</p>
              </article>
              <p className="status waiting" role="status">Помощник готовит ответ…</p>
            </>
          )}
          <div ref={end} />
        </div>
      </section>

      <footer className="composer">
        {(sendError || historyError) && (
          <div className="error-notice" role="alert">
            {sendError && <p>{sendError}</p>}
            {historyError && <p>{historyError}</p>}
            {historyError && (
              <button className="retry-button" disabled={loading || sending} onClick={() => refreshHistory()}>
                {loading ? "Загрузка…" : "Повторить загрузку истории"}
              </button>
            )}
          </div>
        )}
        <form onSubmit={handleSubmit}>
          <label htmlFor="message">Ваше сообщение</label>
          <div className="input-row">
            <textarea
              id="message"
              ref={input}
              rows={2}
              placeholder="Напишите вопрос…"
              value={draft}
              disabled={sending}
              aria-describedby="input-hint character-count"
              onChange={(event) => setDraft(Array.from(event.target.value).slice(0, 2000).join(""))}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                  event.preventDefault();
                  event.currentTarget.form.requestSubmit();
                }
              }}
            />
            <button className="send-button" type="submit" disabled={!canSend} aria-label="Отправить сообщение">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
                <path d="m5 12 7-7 7 7M12 5v15" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          </div>
          <div className="input-meta">
            <span id="input-hint">Enter — отправить · Shift+Enter — новая строка</span>
            <span id="character-count">{length} / 2000</span>
          </div>
        </form>
      </footer>
    </main>
  );
}
