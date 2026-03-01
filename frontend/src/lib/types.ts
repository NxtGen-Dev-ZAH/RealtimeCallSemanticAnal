export type AnalysisStatus = 'pending' | 'processing' | 'completed' | 'failed' | 'unknown';

export interface UploadResponse {
  status?: string;
  message?: string;
  call_id: string;
  filename: string;
  size: number;
}

export interface AnalyzeResponse {
  message: string;
  call_id: string;
  status: AnalysisStatus;
}

export interface StatusResponse {
  call_id: string;
  status: AnalysisStatus;
  progress: number;
}

export interface SentimentScore {
  timestamp: number;
  score: number;
  label?: string;
}

export interface KeyPhrase {
  phrase: string;
  score: number;
}

export interface KeyPhrases {
  positive: KeyPhrase[];
  negative: KeyPhrase[];
}

export interface CallSummary {
  avg_sentiment: number;
  total_duration: number;
  participants: number;
}

export interface CallResults {
  call_id: string;
  sentiment_scores: SentimentScore[];
  emotions: Record<string, number>;
  sale_probability: number;
  key_phrases: KeyPhrases;
  summary: CallSummary;
}

export interface CallHistoryItem {
  call_id: string;
  filename: string;
  timestamp: string;
  duration: number;
  avg_sentiment: number;
  sale_probability: number;
  participants: number;
  status: AnalysisStatus | string;
}
