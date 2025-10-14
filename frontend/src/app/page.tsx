'use client';

import { useState } from 'react';
import UploadForm from '@/components/UploadForm';
import AnalysisDashboard from '@/components/AnalysisDashboard';
import { apiService } from '@/lib/api';
import toast from 'react-hot-toast';
import { Mic, Brain, TrendingUp, Shield, Zap, Sparkles } from 'lucide-react';

export default function Home() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStatus, setAnalysisStatus] = useState(null);

  const handleUploadSuccess = (fileData: any) => {
    setUploadedFile(fileData);
    setAnalysisResults(null);
    setAnalysisStatus(null);
    toast.success('File uploaded successfully!');
  };

  const handleUploadError = (error: any) => {
    toast.error(`Upload failed: ${error.message}`);
  };

  const handleAnalyze = async () => {
    if (!uploadedFile) {
      toast.error('Please upload a file first');
      return;
    }

    setIsAnalyzing(true);
    setAnalysisStatus({ status: 'processing', progress: 0 });

    try {
      // Start analysis
      const analysisResponse = await apiService.analyzeCall(uploadedFile.call_id);
      console.log('Analysis started:', analysisResponse);

      // Poll for status updates
      const pollStatus = async () => {
        try {
          const statusResponse = await apiService.getStatus(uploadedFile.call_id);
          setAnalysisStatus(statusResponse);

          if (statusResponse.status === 'processing') {
            // Continue polling
            setTimeout(pollStatus, 2000);
          } else if (statusResponse.status === 'completed') {
            // Fetch results
            const results = await apiService.getResults(uploadedFile.call_id);
            setAnalysisResults(results);
            setIsAnalyzing(false);
            toast.success('Analysis completed successfully!');
          } else if (statusResponse.status === 'failed') {
            setIsAnalyzing(false);
            toast.error('Analysis failed. Please try again.');
          }
        } catch (error) {
          console.error('Error polling status:', error);
          setIsAnalyzing(false);
          toast.error('Error checking analysis status');
        }
      };

      // Start polling
      setTimeout(pollStatus, 1000);
    } catch (error: any) {
      console.error('Error starting analysis:', error);
      setIsAnalyzing(false);
      setAnalysisStatus(null);
      toast.error(`Analysis failed: ${error.message}`);
    }
  };

  const handleReset = () => {
    setUploadedFile(null);
    setAnalysisResults(null);
    setIsAnalyzing(false);
    setAnalysisStatus(null);
  };

  return (
    <div className="pt-20">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-purple-600/20 to-pink-600/20 blur-3xl"></div>
        <div className="relative container mx-auto px-6 py-20 text-center">
          <div className="max-w-4xl mx-auto">
            <div className="inline-flex items-center space-x-2 bg-white/10 backdrop-blur-sm border border-white/20 rounded-full px-4 py-2 mb-8">
              <Sparkles className="h-4 w-4 text-yellow-400" />
              <span className="text-white/90 text-sm font-medium">AI-Powered Conversation Intelligence</span>
            </div>
            
            <h1 className="text-5xl md:text-7xl font-bold mb-6">
              <span className="bg-gradient-to-r from-white via-purple-200 to-pink-200 bg-clip-text text-transparent">
                Call Analysis
              </span>
              <br />
              <span className="bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                Revolution
              </span>
            </h1>
            
            <p className="text-xl text-white/80 max-w-3xl mx-auto mb-12 leading-relaxed">
              Transform your call center operations with advanced AI that analyzes sentiment, 
              emotions, and sales potential in real-time. Get actionable insights that drive results.
            </p>

            {/* Feature Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16">
              <div className="group bg-white/10 backdrop-blur-sm border border-white/20 rounded-2xl p-6 hover:bg-white/15 transition-all duration-300">
                <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-xl flex items-center justify-center mb-4 mx-auto group-hover:scale-110 transition-transform duration-300">
                  <Mic className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-white font-semibold mb-2">Speech Recognition</h3>
                <p className="text-white/70 text-sm">Advanced Whisper AI for accurate transcription</p>
              </div>
              
              <div className="group bg-white/10 backdrop-blur-sm border border-white/20 rounded-2xl p-6 hover:bg-white/15 transition-all duration-300">
                <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl flex items-center justify-center mb-4 mx-auto group-hover:scale-110 transition-transform duration-300">
                  <Brain className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-white font-semibold mb-2">Sentiment Analysis</h3>
                <p className="text-white/70 text-sm">BERT-powered emotion and sentiment detection</p>
              </div>
              
              <div className="group bg-white/10 backdrop-blur-sm border border-white/20 rounded-2xl p-6 hover:bg-white/15 transition-all duration-300">
                <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-emerald-500 rounded-xl flex items-center justify-center mb-4 mx-auto group-hover:scale-110 transition-transform duration-300">
                  <TrendingUp className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-white font-semibold mb-2">Sales Prediction</h3>
                <p className="text-white/70 text-sm">XGBoost ML for conversion probability</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-6 pb-20">
        <div className="space-y-12">

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Upload Section */}
            <div className="bg-white/10 backdrop-blur-sm border border-white/20 rounded-3xl p-8">
              <div className="flex items-center space-x-3 mb-6">
                <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-xl flex items-center justify-center">
                  <Mic className="h-5 w-5 text-white" />
                </div>
                <h2 className="text-2xl font-bold text-white">
                  Upload Audio File
                </h2>
              </div>
          <UploadForm
            onUploadSuccess={handleUploadSuccess}
            onUploadError={handleUploadError}
            disabled={isAnalyzing}
          />
          
              {uploadedFile && (
                <div className="mt-6 p-6 bg-green-500/20 backdrop-blur-sm border border-green-400/30 rounded-2xl">
                  <div className="flex items-center space-x-2 mb-3">
                    <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center">
                      <Shield className="h-3 w-3 text-white" />
                    </div>
                    <h3 className="font-semibold text-green-200">File Uploaded Successfully</h3>
                  </div>
                  <p className="text-sm text-green-300/80 mb-2">
                    {uploadedFile.filename} ({uploadedFile.size} bytes)
                  </p>
                  <p className="text-sm text-green-300/80">
                    Call ID: {uploadedFile.call_id}
                  </p>
                </div>
              )}
            </div>

            {/* Analysis Section */}
            <div className="bg-white/10 backdrop-blur-sm border border-white/20 rounded-3xl p-8">
              <div className="flex items-center space-x-3 mb-6">
                <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl flex items-center justify-center">
                  <Zap className="h-5 w-5 text-white" />
                </div>
                <h2 className="text-2xl font-bold text-white">
                  Run Analysis
                </h2>
              </div>
          
              {!uploadedFile ? (
                <div className="text-center py-12">
                  <div className="w-16 h-16 bg-white/10 rounded-2xl flex items-center justify-center mx-auto mb-4">
                    <Brain className="h-8 w-8 text-white/50" />
                  </div>
                  <p className="text-white/60">Please upload an audio file first to start analysis.</p>
                </div>
              ) : (
                <div className="space-y-6">
                  <button
                    onClick={handleAnalyze}
                    disabled={isAnalyzing}
                    className="group relative w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-600 text-white font-semibold py-4 px-6 rounded-2xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed overflow-hidden"
                  >
                    <div className="absolute inset-0 bg-gradient-to-r from-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                    <span className="relative flex items-center justify-center space-x-2">
                      {isAnalyzing ? (
                        <>
                          <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                          <span>Analyzing...</span>
                        </>
                      ) : (
                        <>
                          <Zap className="h-5 w-5" />
                          <span>Run Analysis</span>
                        </>
                      )}
                    </span>
                  </button>

                  {analysisStatus && (
                    <div className="p-6 bg-blue-500/20 backdrop-blur-sm border border-blue-400/30 rounded-2xl">
                      <div className="flex items-center space-x-2 mb-3">
                        <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center">
                          <Brain className="h-3 w-3 text-white" />
                        </div>
                        <h3 className="font-semibold text-blue-200">Analysis Status</h3>
                      </div>
                      <p className="text-sm text-blue-300/80 mb-3">
                        Status: <span className="capitalize font-medium">{analysisStatus.status}</span>
                      </p>
                      {analysisStatus.progress !== undefined && (
                        <div className="space-y-2">
                          <div className="w-full bg-blue-900/30 rounded-full h-3 overflow-hidden">
                            <div 
                              className="bg-gradient-to-r from-blue-500 to-cyan-500 h-3 rounded-full transition-all duration-500 ease-out"
                              style={{ width: `${analysisStatus.progress}%` }}
                            ></div>
                          </div>
                          <p className="text-xs text-blue-300/80 text-center">
                            {analysisStatus.progress}% complete
                          </p>
                        </div>
                      )}
                    </div>
                  )}

                  <button
                    onClick={handleReset}
                    className="w-full bg-white/10 hover:bg-white/20 backdrop-blur-sm border border-white/20 hover:border-white/30 text-white font-medium py-3 px-6 rounded-2xl transition-all duration-300"
                  >
                    Reset
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Results Section */}
          {analysisResults && (
            <div className="bg-white/10 backdrop-blur-sm border border-white/20 rounded-3xl p-8">
              <div className="flex items-center space-x-3 mb-8">
                <div className="w-10 h-10 bg-gradient-to-r from-green-500 to-emerald-500 rounded-xl flex items-center justify-center">
                  <TrendingUp className="h-5 w-5 text-white" />
                </div>
                <h2 className="text-2xl font-bold text-white">
                  Analysis Results
                </h2>
              </div>
              <AnalysisDashboard results={analysisResults} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
