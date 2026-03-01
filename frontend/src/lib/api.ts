import axios from 'axios';
import type {
  AnalyzeResponse,
  AnalysisStatus,
  CallHistoryItem,
  CallResults,
  KeyPhrase,
  StatusResponse,
  UploadResponse,
} from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || '/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
});

const toNumber = (value: unknown, fallback = 0): number => {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
};

const toStatus = (value: unknown): AnalysisStatus => {
  if (value === 'pending' || value === 'processing' || value === 'completed' || value === 'failed') {
    return value;
  }
  return 'unknown';
};

const normalizePhrase = (item: unknown): KeyPhrase | null => {
  if (typeof item === 'string') {
    const phrase = item.trim();
    return phrase ? { phrase, score: 0 } : null;
  }

  if (!item || typeof item !== 'object') return null;
  const value = item as Record<string, unknown>;
  const phrase = typeof value.phrase === 'string' ? value.phrase.trim() : '';
  if (!phrase) return null;
  return {
    phrase,
    score: toNumber(value.score, toNumber(value.sentiment_score, 0)),
  };
};

const normalizeErrorMessage = (error: unknown): string => {
  if (axios.isAxiosError(error)) {
    const data = error.response?.data;

    if (typeof data === 'string' && data.trim()) {
      return data;
    }

    if (data && typeof data === 'object') {
      const responseObject = data as Record<string, unknown>;
      if (typeof responseObject.detail === 'string' && responseObject.detail.trim()) {
        return responseObject.detail;
      }
      if (typeof responseObject.message === 'string' && responseObject.message.trim()) {
        return responseObject.message;
      }
    }

    return error.message || 'Request failed';
  }

  if (error instanceof Error) return error.message;
  return 'Unexpected API error';
};

const toError = (error: unknown): Error => new Error(normalizeErrorMessage(error));

const normalizeResults = (data: unknown): CallResults => {
  const payload = (data && typeof data === 'object' ? data : {}) as Record<string, unknown>;

  const rawScores = Array.isArray(payload.sentiment_scores) ? payload.sentiment_scores : [];
  const sentiment_scores = rawScores.flatMap((item) => {
    if (!item || typeof item !== 'object') return [];
    const value = item as Record<string, unknown>;
    const label = typeof value.label === 'string' ? value.label : undefined;
    return [
      {
        timestamp: toNumber(value.timestamp, toNumber(value.start_time, 0)),
        score: toNumber(value.score, 0),
        label,
      },
    ];
  });

  const rawEmotions =
    payload.emotions && typeof payload.emotions === 'object'
      ? (payload.emotions as Record<string, unknown>)
      : {};
  const emotions = Object.fromEntries(
    Object.entries(rawEmotions).map(([key, value]) => [key, toNumber(value, 0)]),
  );

  const rawKeyPhrases =
    payload.key_phrases && typeof payload.key_phrases === 'object'
      ? (payload.key_phrases as Record<string, unknown>)
      : {};

  const positiveRaw = Array.isArray(rawKeyPhrases.positive) ? rawKeyPhrases.positive : [];
  const negativeRaw = Array.isArray(rawKeyPhrases.negative) ? rawKeyPhrases.negative : [];

  const key_phrases = {
    positive: positiveRaw.map(normalizePhrase).filter((item): item is KeyPhrase => item !== null),
    negative: negativeRaw.map(normalizePhrase).filter((item): item is KeyPhrase => item !== null),
  };

  const rawSummary =
    payload.summary && typeof payload.summary === 'object'
      ? (payload.summary as Record<string, unknown>)
      : {};

  return {
    call_id: typeof payload.call_id === 'string' ? payload.call_id : '',
    sentiment_scores,
    emotions,
    sale_probability: toNumber(payload.sale_probability, 0),
    key_phrases,
    summary: {
      avg_sentiment: toNumber(rawSummary.avg_sentiment, 0),
      total_duration: toNumber(rawSummary.total_duration, 0),
      participants: toNumber(rawSummary.participants, 0),
    },
  };
};

const normalizeHistory = (data: unknown): CallHistoryItem[] => {
  if (!Array.isArray(data)) return [];

  return data.map((item) => {
    const payload = (item && typeof item === 'object' ? item : {}) as Record<string, unknown>;
    return {
      call_id: typeof payload.call_id === 'string' ? payload.call_id : '',
      filename: typeof payload.filename === 'string' ? payload.filename : 'unknown',
      timestamp: typeof payload.timestamp === 'string' ? payload.timestamp : new Date().toISOString(),
      duration: toNumber(payload.duration, 0),
      avg_sentiment: toNumber(payload.avg_sentiment, 0),
      sale_probability: toNumber(payload.sale_probability, 0),
      participants: toNumber(payload.participants, 0),
      status: toStatus(payload.status),
    };
  });
};

export const apiService = {
  async uploadAudio(file: File): Promise<UploadResponse> {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await apiClient.post('/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const data = response.data as Record<string, unknown>;
      const callId = typeof data.call_id === 'string' ? data.call_id : '';
      if (!callId) {
        throw new Error('Upload succeeded but call_id was missing in response');
      }

      return {
        status: typeof data.status === 'string' ? data.status : undefined,
        message: typeof data.message === 'string' ? data.message : undefined,
        call_id: callId,
        filename: typeof data.filename === 'string' ? data.filename : file.name,
        size: toNumber(data.size, file.size),
      };
    } catch (error) {
      throw toError(error);
    }
  },

  async analyzeCall(callId: string): Promise<AnalyzeResponse> {
    try {
      const response = await apiClient.post('/analyze', { call_id: callId });
      const data = (response.data || {}) as Record<string, unknown>;

      return {
        message: typeof data.message === 'string' ? data.message : 'Analysis started',
        call_id: typeof data.call_id === 'string' ? data.call_id : callId,
        status: toStatus(data.status),
      };
    } catch (error) {
      throw toError(error);
    }
  },

  async getStatus(callId: string): Promise<StatusResponse> {
    try {
      const response = await apiClient.get(`/status/${encodeURIComponent(callId)}`);
      const data = (response.data || {}) as Record<string, unknown>;

      return {
        call_id: typeof data.call_id === 'string' ? data.call_id : callId,
        status: toStatus(data.status),
        progress: toNumber(data.progress, 0),
      };
    } catch (error) {
      throw toError(error);
    }
  },

  async getResults(callId: string): Promise<CallResults> {
    try {
      const response = await apiClient.get(`/results/${encodeURIComponent(callId)}`);
      const normalized = normalizeResults(response.data);

      return {
        ...normalized,
        call_id: normalized.call_id || callId,
      };
    } catch (error) {
      throw toError(error);
    }
  },

  async getHistory(): Promise<CallHistoryItem[]> {
    try {
      const response = await apiClient.get('/history');
      return normalizeHistory(response.data);
    } catch (error) {
      throw toError(error);
    }
  },

  async exportJSON(callId: string): Promise<Blob> {
    try {
      const response = await apiClient.get(`/export/${encodeURIComponent(callId)}`, {
        responseType: 'blob',
      });
      return response.data as Blob;
    } catch (error) {
      throw toError(error);
    }
  },

  async exportPDF(callId: string): Promise<Blob> {
    try {
      const response = await apiClient.get(`/export/${encodeURIComponent(callId)}/pdf`, {
        responseType: 'blob',
      });
      return response.data as Blob;
    } catch (error) {
      throw toError(error);
    }
  },

  async exportCSV(callId: string): Promise<Blob> {
    try {
      const response = await apiClient.get(`/export/${encodeURIComponent(callId)}/csv`, {
        responseType: 'blob',
      });
      return response.data as Blob;
    } catch (error) {
      throw toError(error);
    }
  },
};
