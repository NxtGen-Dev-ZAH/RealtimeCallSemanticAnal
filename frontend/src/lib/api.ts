/**
 * API service for the Call Analysis System
 */

import type { 
  UploadResponse,
  AnalyzeResponse,
  StatusResponse, 
  CallResults, 
  CallHistoryItem 
} from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

class ApiService {
  /**
   * Upload an audio file for analysis
   */
  async uploadAudio(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/api/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Upload failed');
    }

    return response.json();
  }

  /**
   * Start analysis for an uploaded call
   */
  async analyzeCall(callId: string): Promise<AnalyzeResponse> {
    const response = await fetch(`${API_BASE_URL}/api/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ call_id: callId }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Analysis failed');
    }

    return response.json();
  }

  /**
   * Get analysis status for a call
   */
  async getStatus(callId: string): Promise<StatusResponse> {
    const response = await fetch(`${API_BASE_URL}/api/status/${callId}`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to get status');
    }

    return response.json();
  }

  /**
   * Get analysis results for a call
   */
  async getResults(callId: string): Promise<CallResults> {
    const response = await fetch(`${API_BASE_URL}/api/results/${callId}`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to get results');
    }

    return response.json();
  }

  /**
   * Get call history
   */
  async getHistory(): Promise<CallHistoryItem[]> {
    const response = await fetch(`${API_BASE_URL}/api/history`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to get history');
    }

    return response.json();
  }
}

export const apiService = new ApiService();
