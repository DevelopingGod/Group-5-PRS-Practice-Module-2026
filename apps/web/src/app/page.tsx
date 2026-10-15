import { ScanFlow } from "@/components/scan-flow";

export default function Home() {
  return (
    <>
      <header className="bg-ink text-light">
        <div className="mx-auto flex max-w-xl items-center px-4 py-3">
          <span className="font-heading text-xl font-bold">Bin Buddy</span>
        </div>
      </header>
      <main className="mx-auto w-full max-w-xl flex-1 px-4 py-6">
        <ScanFlow />
      </main>
    </>
  );
}
