import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Toaster } from 'react-hot-toast';
import Navbar from '@/components/Navbar';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Call Analysis System | AI-Powered Conversation Intelligence',
  description: 'Advanced multimodal sentiment analysis for telephonic conversations with real-time insights and sales prediction',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={inter.className}>
        <div className="min-h-screen relative bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
          {/* Decorative gradient overlay (removed problematic inline SVG to avoid JSX parse errors) */}
          <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-blue-900/20 via-transparent to-slate-900/20"></div>
          <Navbar />
          <main className="relative z-10">
            {children}
          </main>
          <Toaster 
            position="bottom-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#363636',
                color: '#fff',
              },
              success: {
                duration: 3000,
                iconTheme: {
                  primary: '#22c55e',
                  secondary: '#fff',
                },
              },
              error: {
                duration: 5000,
                iconTheme: {
                  primary: '#ef4444',
                  secondary: '#fff',
                },
              },
            }}
          />
        </div>
      </body>
    </html>
  );
}
