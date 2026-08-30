import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "SatQuery AI — Ask Your Satellite Images Anything",
  description:
    "An interactive vision-language assistant for multimodal remote sensing image analysis through natural language queries. Built for ISRO SIH26167.",
  keywords: [
    "SatQuery AI",
    "Remote Sensing",
    "Satellite Imagery",
    "Vision-Language Model",
    "ISRO",
    "SIH26167",
    "Geospatial AI",
    "GIS",
  ],
  authors: [{ name: "SatQuery AI Team" }],
  icons: {
    icon: "/logo.svg",
  },
  openGraph: {
    title: "SatQuery AI",
    description: "Ask Your Satellite Images Anything — Natural Language Remote Sensing Analysis",
    siteName: "SatQuery AI",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "SatQuery AI",
    description: "Ask Your Satellite Images Anything — Natural Language Remote Sensing Analysis",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        {children}
        <Toaster />
      </body>
    </html>
  );
}
