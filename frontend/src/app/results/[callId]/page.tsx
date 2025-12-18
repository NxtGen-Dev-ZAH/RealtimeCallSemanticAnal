'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import AnalysisDashboard from '@/components/AnalysisDashboard';
import { apiService } from '@/lib/api';
import type { CallResults } from '@/lib/types';
import { ArrowLeft, Loader2 } from 'lucide-react';
import toast from 'react-hot-toast';

interface ResultsPageProps {
  params: {
    callId: string;
  };
}

export default function ResultsPage({ params }: ResultsPageProps) {
  const { callId } = params;
  const router = useRouter();
  const [results, setResults] = useState<CallResults | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchResults = async () => {
      try {
        const data = await apiService.getResults(callId);
        setResults(data);
      } catch (error: any) {
        toast.error(error?.message ?? 'Failed to load results');
      } finally {
        setLoading(false);
      }
    };

    fetchResults();
  }, [callId]);

  return (
    <div className="pt-24 pb-16 container mx-auto px-6">
      <button
        onClick={() => router.back()}
        className="mb-6 inline-flex items-center space-x-2 text-white/80 hover:text-white transition-colors text-sm"
      >
        <ArrowLeft className="h-4 w-4" />
        <span>Back</span>
      </button>

      {loading && (
        <div className="flex items-center justify-center h-64">
          <Loader2 className="h-10 w-10 animate-spin text-white/80" />
        </div>
      )}

      {!loading && !results && (
        <div className="card text-center py-12">
          <p className="text-white/80 mb-2">No results found for this call.</p>
          <p className="text-white/60 text-sm">The analysis may have failed or the call ID is invalid.</p>
        </div>
      )}

      {results && (
        <div className="bg-white/10 backdrop-blur-sm border border-white/20 rounded-3xl p-8">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-2xl font-bold text-white mb-1">Analysis Results</h1>
              <p className="text-sm text-white/70">Call ID: {results.call_id}</p>
            </div>
          </div>
          <AnalysisDashboard results={results} />
        </div>
      )}
    </div>
  );
}
