/**
 * Type definitions for the Call Analysis System API
 */

// Upload response from POST /api/upload
export interface UploadResponse {
  status: string;
  message: string;
  call_id: string;
  filename: string;
  size: number;
}

// Analyze response from POST /api/analyze
export interface AnalyzeResponse {
  message: string;
  call_id: string;
  status: string;
}

// Call status types
export type CallStatus = 'processing' | 'completed' | 'failed' | 'unknown';

// Status response from GET /api/status/{call_id}
export interface StatusResponse {
  call_id: string;
  status: CallStatus;
  progress: number;
}

// Sentiment score data point
export interface SentimentScore {
  timestamp: number;
  score: number;
}

// Emotion distribution
export interface Emotions {
  [emotion: string]: number;
}

// Key phrases
export interface KeyPhrases {
  positive: string[];
  negative: string[];
}

// Summary information
export interface Summary {
  avg_sentiment: number;
  total_duration: number;
  participants: number;
}

// Call results from GET /api/results/{call_id}
export interface CallResults {
  call_id: string;
  sentiment_scores: SentimentScore[];
  emotions: Emotions;
  sale_probability: number;
  key_phrases: KeyPhrases;
  summary: Summary;
}

// Call history item from GET /api/history
export interface CallHistoryItem {
  _id: string;
  call_id: string;
  filename: string;
  timestamp: string;
  duration: number;
  avg_sentiment: number;
  sale_probability: number;
  participants: number;
  status: string;
}
