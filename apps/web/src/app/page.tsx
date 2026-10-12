import { CONTRACTS_VERSION, QUESTION_BANK } from "@/contracts/constants";

// Placeholder home page. WS4 owns the real capture -> question -> verdict flow.
export default function Home() {
  return (
    <main className="mx-auto max-w-xl p-6">
      <h1 className="text-2xl font-semibold">Bin Buddy</h1>
      <p className="mt-2">Scaffold running against contracts v{CONTRACTS_VERSION}.</p>
      <ul className="mt-4 list-disc pl-6">
        {Object.values(QUESTION_BANK).map((q) => (
          <li key={q.text}>{q.text}</li>
        ))}
      </ul>
    </main>
  );
}
