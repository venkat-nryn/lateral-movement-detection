import type { Metadata } from "next";
import "./globals.css";
import { ReplayProvider } from "@/lib/replay";
import { Sidebar } from "@/components/Sidebar";
import { ReplayBar } from "@/components/ReplayBar";

export const metadata: Metadata = {
  title: "LMD · Lateral Movement SOC",
  description: "Analyst console for the real-LANL lateral-movement detection pipeline",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="h-screen overflow-hidden font-sans text-[13px]">
        <ReplayProvider>
          <div className="flex h-full">
            <Sidebar />
            <div className="flex min-w-0 flex-1 flex-col">
              <ReplayBar />
              <main className="min-h-0 flex-1 overflow-y-auto">
                <div className="mx-auto max-w-[1680px] animate-fadeIn p-5">{children}</div>
              </main>
            </div>
          </div>
        </ReplayProvider>
      </body>
    </html>
  );
}
