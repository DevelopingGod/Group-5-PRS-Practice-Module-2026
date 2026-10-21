import { BottomNav } from "@/components/bottom-nav";
import { ScanFlow } from "@/components/scan-flow";
import { SiteHeader } from "@/components/site-header";

export default function Home() {
  return (
    <>
      <SiteHeader showProfile />
      <main className="mx-auto w-full max-w-xl flex-1 px-4 py-6">
        <ScanFlow />
      </main>
      <BottomNav />
    </>
  );
}
