'use client';

import { useState } from 'react';
import SentimentChart from './SentimentChart';
import EmotionChart from './EmotionChart';
import SaleGauge from './SaleGauge';
import KeyPhrases from './KeyPhrases';
import { TrendingUp, TrendingDown, Users, Clock, Download, FileText, FileSpreadsheet } from 'lucide-react';
import toast from 'react-hot-toast';
import type { CallResults } from '@/lib/types';

interface AnalysisDashboardProps {
  results: CallResults;
}

const AnalysisDashboard = ({ results }: AnalysisDashboardProps) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [isExporting, setIsExporting] = useState(false);

  const handleExportPDF = async () => {
    setIsExporting(true);
    try {
      const response = await fetch(`/api/export/${results.call_id}/pdf`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${results.call_id}_report.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        toast.success('PDF report downloaded successfully!');
      } else {
        toast.error('Failed to export PDF report');
      }
    } catch (error) {
      toast.error('Error exporting PDF report');
    } finally {
      setIsExporting(false);
    }
  };

  const handleExportJSON = async () => {
    setIsExporting(true);
    try {
      const response = await fetch(`/api/export/${results.call_id}`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${results.call_id}_analysis.json`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        toast.success('JSON data downloaded successfully!');
      } else {
        toast.error('Failed to export JSON data');
      }
    } catch (error) {
      toast.error('Error exporting JSON data');
    } finally {
      setIsExporting(false);
    }
  };

  const handleExportCSV = async () => {
    setIsExporting(true);
    try {
      const response = await fetch(`/api/export/${results.call_id}/csv`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${results.call_id}_analysis.csv`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        toast.success('CSV data downloaded successfully!');
      } else {
        toast.error('Failed to export CSV data');
      }
    } catch (error) {
      toast.error('Error exporting CSV data');
    } finally {
      setIsExporting(false);
    }
  };

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'sentiment', label: 'Sentiment Analysis' },
    { id: 'emotions', label: 'Emotion Detection' },
    { id: 'phrases', label: 'Key Phrases' },
  ];

  const getSentimentLabel = (score: number) => {
    if (score >= 0.6) return 'Positive';
    if (score >= 0.4) return 'Neutral';
    return 'Negative';
  };

  const getSentimentColor = (score: number) => {
    if (score >= 0.6) return 'text-green-600';
    if (score >= 0.4) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Average Sentiment</p>
              <p className={`text-2xl font-bold ${getSentimentColor(results.summary.avg_sentiment)}`}>
                {getSentimentLabel(results.summary.avg_sentiment)}
              </p>
              <p className="text-sm text-gray-500">
                Score: {results.summary.avg_sentiment.toFixed(2)}
              </p>
            </div>
            {results.summary.avg_sentiment >= 0.5 ? (
              <TrendingUp className="h-8 w-8 text-green-600" />
            ) : (
              <TrendingDown className="h-8 w-8 text-red-600" />
            )}
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Sale Probability</p>
              <p className="text-2xl font-bold text-primary-600">
                {(results.sale_probability * 100).toFixed(1)}%
              </p>
            </div>
            <TrendingUp className="h-8 w-8 text-primary-600" />
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Duration</p>
              <p className="text-2xl font-bold text-gray-900">
                {Math.floor(results.summary.total_duration / 60)}m {results.summary.total_duration % 60}s
              </p>
            </div>
            <Clock className="h-8 w-8 text-gray-600" />
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Participants</p>
              <p className="text-2xl font-bold text-gray-900">
                {results.summary.participants}
              </p>
            </div>
            <Users className="h-8 w-8 text-gray-600" />
          </div>
        </div>
      </div>

      {/* Header with Export Buttons */}
      <div className="flex justify-between items-center mb-6">
        <div className="border-b border-gray-200 flex-1">
          <nav className="-mb-px flex space-x-8">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-primary-500 text-primary-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
        
        {/* Export Buttons */}
        <div className="flex space-x-2 ml-6">
          <button
            onClick={handleExportPDF}
            disabled={isExporting}
            className="btn-secondary flex items-center space-x-2 disabled:opacity-50"
          >
            <FileText className="h-4 w-4" />
            <span>PDF</span>
          </button>
          <button
            onClick={handleExportCSV}
            disabled={isExporting}
            className="btn-secondary flex items-center space-x-2 disabled:opacity-50"
          >
            <FileSpreadsheet className="h-4 w-4" />
            <span>CSV</span>
          </button>
          <button
            onClick={handleExportJSON}
            disabled={isExporting}
            className="btn-secondary flex items-center space-x-2 disabled:opacity-50"
          >
            <Download className="h-4 w-4" />
            <span>JSON</span>
          </button>
        </div>
      </div>

      {/* Tab Content */}
      <div className="space-y-6">
        {activeTab === 'overview' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Sentiment Over Time</h3>
              <SentimentChart data={results.sentiment_scores} />
            </div>
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Emotion Distribution</h3>
              <EmotionChart data={results.emotions} />
            </div>
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Sale Probability</h3>
              <SaleGauge value={results.sale_probability} />
            </div>
          </div>
        )}

        {activeTab === 'sentiment' && (
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Sentiment Analysis</h3>
            <SentimentChart data={results.sentiment_scores} />
          </div>
        )}

        {activeTab === 'emotions' && (
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Emotion Detection</h3>
            <EmotionChart data={results.emotions} />
          </div>
        )}

        {activeTab === 'phrases' && (
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Key Phrases</h3>
            <KeyPhrases phrases={results.key_phrases} />
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalysisDashboard;
