import type { Metadata } from "next";
import "./globals.css";

// Typography and visual design are WS4's call; the scaffold uses the system font stack.
export const metadata: Metadata = {
  title: "Bin Buddy",
  description: "Which bin does this go in? An explainable recycling assistant for Singapore.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
