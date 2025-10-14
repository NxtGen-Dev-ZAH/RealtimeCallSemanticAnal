'use client';

import { useState, useRef } from 'react';
import { Upload, FileAudio, X, Cloud, Sparkles } from 'lucide-react';
import { apiService } from '@/lib/api';
import toast from 'react-hot-toast';

interface UploadFormProps {
  onUploadSuccess: (fileData: any) => void;
  onUploadError: (error: any) => void;
  disabled?: boolean;
}

const UploadForm = ({ onUploadSuccess, onUploadError, disabled }: UploadFormProps) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const allowedTypes = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/m4a', 'audio/mp4'];
  const maxSize = 100 * 1024 * 1024; // 100MB

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!allowedTypes.includes(file.type)) {
      toast.error('Please select a valid audio file (.wav, .mp3, .m4a)');
      return;
    }

    // Validate file size
    if (file.size > maxSize) {
      toast.error('File size must be less than 100MB');
      return;
    }

    setSelectedFile(file);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setUploadProgress(0);

    try {
      const response = await apiService.uploadAudio(selectedFile);
      onUploadSuccess({
        ...response,
        filename: selectedFile.name,
        size: selectedFile.size,
      });
    } catch (error: any) {
      onUploadError(error);
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="space-y-6">
      {/* File Input */}
      <div className="relative group">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-2xl blur opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
        <div className="relative border-2 border-dashed border-white/30 rounded-2xl p-8 text-center hover:border-white/50 transition-all duration-300 bg-white/5 backdrop-blur-sm">
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.mp3,.m4a"
            onChange={handleFileSelect}
            disabled={disabled || isUploading}
            className="hidden"
          />
          
          {!selectedFile ? (
            <div
              onClick={() => fileInputRef.current?.click()}
              className="cursor-pointer group/upload"
            >
              <div className="relative w-20 h-20 mx-auto mb-6">
                <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500 rounded-2xl blur opacity-30 group-hover/upload:opacity-50 transition-opacity duration-300"></div>
                <div className="relative w-full h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-2xl flex items-center justify-center group-hover/upload:scale-110 transition-transform duration-300">
                  <Cloud className="h-10 w-10 text-white" />
                </div>
              </div>
              <p className="text-xl font-semibold text-white mb-3">
                Drop your audio file here
              </p>
              <p className="text-white/70 mb-4">
                or click to browse files
              </p>
              <div className="inline-flex items-center space-x-2 bg-white/10 backdrop-blur-sm border border-white/20 rounded-full px-4 py-2">
                <Sparkles className="h-4 w-4 text-yellow-400" />
                <span className="text-white/90 text-sm">
                  Supports .wav, .mp3, .m4a up to 100MB
                </span>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-between p-6 bg-white/10 backdrop-blur-sm border border-white/20 rounded-2xl">
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-emerald-500 rounded-xl flex items-center justify-center">
                  <FileAudio className="h-6 w-6 text-white" />
                </div>
                <div className="text-left">
                  <p className="font-semibold text-white">{selectedFile.name}</p>
                  <p className="text-sm text-white/70">{formatFileSize(selectedFile.size)}</p>
                </div>
              </div>
              <button
                onClick={handleRemoveFile}
                disabled={disabled || isUploading}
                className="w-8 h-8 bg-red-500/20 hover:bg-red-500/30 border border-red-400/30 rounded-lg flex items-center justify-center text-red-300 hover:text-red-200 disabled:opacity-50 transition-all duration-200"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Upload Button */}
      {selectedFile && (
        <button
          onClick={handleUpload}
          disabled={disabled || isUploading}
          className="group relative w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:from-gray-600 disabled:to-gray-600 text-white font-semibold py-4 px-6 rounded-2xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed overflow-hidden"
        >
          <div className="absolute inset-0 bg-gradient-to-r from-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
          <span className="relative flex items-center justify-center space-x-2">
            {isUploading ? (
              <>
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                <span>Uploading...</span>
              </>
            ) : (
              <>
                <Upload className="h-5 w-5" />
                <span>Upload File</span>
              </>
            )}
          </span>
        </button>
      )}

      {/* Upload Progress */}
      {isUploading && (
        <div className="space-y-3">
          <div className="flex justify-between text-sm text-white/80">
            <span>Uploading to cloud...</span>
            <span>{uploadProgress}%</span>
          </div>
          <div className="w-full bg-white/10 rounded-full h-3 overflow-hidden">
            <div
              className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-500 ease-out"
              style={{ width: `${uploadProgress}%` }}
            ></div>
          </div>
        </div>
      )}
    </div>
  );
};

export default UploadForm;
