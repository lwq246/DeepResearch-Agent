import ChatWindow from "@/components/ChatWindow";

export default function HomePage() {
  return (
    <main className="relative min-h-screen overflow-hidden px-4 py-8 sm:px-8">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute -left-16 top-12 h-56 w-56 rounded-full bg-accent/20 blur-3xl" />
        <div className="absolute -right-20 bottom-10 h-64 w-64 rounded-full bg-ember/20 blur-3xl" />
      </div>

      <section className="relative mx-auto flex w-full max-w-4xl flex-col gap-5 rounded-3xl border border-ink/10 bg-white/70 p-4 shadow-panel backdrop-blur sm:p-6">
        <header className="animate-rise">
          <p className="text-xs font-semibold uppercase tracking-[0.25em] text-accent">
            Research Copilot
          </p>
          <h1 className="mt-2 font-display text-3xl leading-tight sm:text-4xl">
            ArXiv RAG Agent
          </h1>
          <p className="mt-2 max-w-2xl text-sm text-ink/75 sm:text-base">
            Ask machine-learning paper questions. The agent searches your Qdrant
            corpus first and falls back to web search only when needed.
          </p>
        </header>

        <ChatWindow />
      </section>
    </main>
  );
}
