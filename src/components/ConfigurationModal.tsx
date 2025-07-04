'use client';

import React, { useState } from 'react';
import { useLanguage } from '@/contexts/LanguageContext';
import UserSelector from './UserSelector';
interface ConfigurationModalProps {
  isOpen: boolean;
  onClose: () => void;
  
  // Repository input
  repositoryInput: string;
  
  // Language selection
  selectedLanguage: string;
  setSelectedLanguage: (value: string) => void;
  
  // Wiki type options
  isComprehensiveView: boolean;
  setIsComprehensiveView: (value: boolean) => void;
  
  // Model selection
  provider: string;
  setProvider: (value: string) => void;
  model: string;
  setModel: (value: string) => void;
  isCustomModel: boolean;
  setIsCustomModel: (value: boolean) => void;
  customModel: string;
  setCustomModel: (value: string) => void;
  
  // Bearer token for backend authentication
  bearerToken: string;
  setBearerToken: (value: string) => void;

  // Platform selection
  selectedPlatform: 'github' | 'gitlab' | 'bitbucket';
  setSelectedPlatform: (value: 'github' | 'gitlab' | 'bitbucket') => void;
  
  // Access token
  accessToken: string;
  setAccessToken: (value: string) => void;
  
  // File filter options
  excludedDirs: string;
  setExcludedDirs: (value: string) => void;
  excludedFiles: string;
  setExcludedFiles: (value: string) => void;
  
  // Form submission
  onSubmit: () => void;
  isSubmitting: boolean;
}

export default function ConfigurationModal({
  isOpen,
  onClose,
  repositoryInput,
  selectedLanguage,
  setSelectedLanguage,
  isComprehensiveView,
  setIsComprehensiveView,
  provider,
  setProvider,
  model,
  setModel,
  isCustomModel,
  setIsCustomModel,
  customModel,
  setCustomModel,
  bearerToken,
  setBearerToken,
  selectedPlatform,
  setSelectedPlatform,
  accessToken,
  setAccessToken,
  excludedDirs,
  setExcludedDirs,
  excludedFiles,
  setExcludedFiles,
  onSubmit,
  isSubmitting
}: ConfigurationModalProps) {
  const { messages: t } = useLanguage();
  
  // Show token section state
  const [showTokenSection, setShowTokenSection] = useState(false);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex min-h-screen items-center justify-center p-4 text-center bg-black/50">
        <div className="relative transform overflow-hidden rounded-lg bg-[var(--card-bg)] text-left shadow-xl transition-all sm:my-8 sm:max-w-2xl sm:w-full">
          {/* Modal header with close button */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--border-color)]">
            <h3 className="text-lg font-medium text-[var(--accent-primary)]">
              <span className="text-[var(--accent-primary)]">{t.form?.configureWiki || 'Configure Wiki'}</span>
            </h3>
            <button
              type="button"
              onClick={onClose}
              className="text-[var(--muted)] hover:text-[var(--foreground)] focus:outline-none transition-colors"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Modal body */}
          <div className="p-6 max-h-[70vh] overflow-y-auto">
            {/* Repository info */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-[var(--foreground)] mb-2">
                {t.form?.repository || 'Repository'}
              </label>
              <div className="bg-[var(--background)]/70 p-3 rounded-md border border-[var(--border-color)] text-sm text-[var(--foreground)]">
                {repositoryInput}
              </div>
            </div>

            {/* Bearer Token */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-[var(--foreground)] mb-2">
                {t.form?.bearerToken || "Bearer Token (for backend authentication)"}
              </label>
              <input
                type="text"
                value={bearerToken}
                onChange={(e) => setBearerToken(e.target.value)}
                placeholder={"Enter your bearer token"}
                className="w-full px-3 py-2 border rounded-md"
                autoComplete='off'
              />
            </div>
            
            {/* Language selection */}
            <div className="mb-4">
              <label htmlFor="language-select" className="block text-sm font-medium text-[var(--foreground)] mb-2">
                {t.form?.wikiLanguage || 'Wiki Language'}
              </label>
              <select
                id="language-select"
                value={selectedLanguage}
                onChange={(e) => setSelectedLanguage(e.target.value)}
                className="input-japanese block w-full px-3 py-2 text-sm rounded-md bg-transparent text-[var(--foreground)] focus:outline-none focus:border-[var(--accent-primary)]"
              >
                <option value="en">English</option>
                <option value="ja">Japanese (日本語)</option>
                <option value="zh">Mandarin (中文)</option>
                <option value="es">Spanish (Español)</option>
                <option value="kr">Korean (한국어)</option>
                <option value="vi">Vietnamese (Tiếng Việt)</option>
              </select>
            </div>
            
            {/* Wiki Type Selector - more compact version */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-[var(--foreground)] mb-2">
                {t.form?.wikiType || 'Wiki Type'}
              </label>
              <div className="flex gap-3">
                <button
                  type="button"
                  onClick={() => setIsComprehensiveView(true)}
                  className={`flex-1 flex items-center justify-between p-2 rounded-md border transition-colors ${
                    isComprehensiveView
                      ? 'bg-[var(--accent-primary)]/10 border-[var(--accent-primary)]/30 text-[var(--accent-primary)]'
                      : 'bg-[var(--background)]/50 border-[var(--border-color)] text-[var(--foreground)] hover:bg-[var(--background)]'
                  }`}
                >
                  <div className="flex items-center">
                    <div className="text-left">
                      <div className="font-medium text-sm">{t.form?.comprehensive || 'Comprehensive'}</div>
                      <div className="text-xs opacity-80">
                        {t.form?.comprehensiveDescription || 'Detailed wiki with structured sections'}
                      </div>
                    </div>
                  </div>
                  {isComprehensiveView && (
                    <div className="ml-2 h-4 w-4 rounded-full bg-[var(--accent-primary)]/20 flex items-center justify-center">
                      <div className="h-2 w-2 rounded-full bg-[var(--accent-primary)]"></div>
                    </div>
                  )}
                </button>
                
                <button
                  type="button"
                  onClick={() => setIsComprehensiveView(false)}
                  className={`flex-1 flex items-center justify-between p-2 rounded-md border transition-colors ${
                    !isComprehensiveView
                      ? 'bg-[var(--accent-primary)]/10 border-[var(--accent-primary)]/30 text-[var(--accent-primary)]'
                      : 'bg-[var(--background)]/50 border-[var(--border-color)] text-[var(--foreground)] hover:bg-[var(--background)]'
                  }`}
                >
                  <div className="flex items-center">
                    <div className="text-left">
                      <div className="font-medium text-sm">{t.form?.concise || 'Concise'}</div>
                      <div className="text-xs opacity-80">
                        {t.form?.conciseDescription || 'Simplified wiki with fewer pages'}
                      </div>
                    </div>
                  </div>
                  {!isComprehensiveView && (
                    <div className="ml-2 h-4 w-4 rounded-full bg-[var(--accent-primary)]/20 flex items-center justify-center">
                      <div className="h-2 w-2 rounded-full bg-[var(--accent-primary)]"></div>
                    </div>
                  )}
                </button>
              </div>
            </div>
            
            {/* Model Selector */}
            <div className="mb-4">
              <UserSelector
                provider={provider}
                setProvider={setProvider}
                model={model}
                setModel={setModel}
                isCustomModel={isCustomModel}
                setIsCustomModel={setIsCustomModel}
                customModel={customModel}
                setCustomModel={setCustomModel}
                showFileFilters={true}
                excludedDirs={excludedDirs}
                setExcludedDirs={setExcludedDirs}
                excludedFiles={excludedFiles}
                setExcludedFiles={setExcludedFiles}
              />
            </div>
            
            {/* Access token section */}
            <div className="mb-4">
              <button
                type="button"
                onClick={() => setShowTokenSection(!showTokenSection)}
                className="text-sm text-[var(--accent-primary)] hover:text-[var(--highlight)] flex items-center transition-colors border-b border-[var(--border-color)] hover:border-[var(--accent-primary)] pb-0.5 mb-2"
              >
                {showTokenSection ? t.form?.hideTokens || 'Hide Access Tokens' : t.form?.addTokens || 'Add Access Tokens for Private Repositories'}
              </button>
              
              {showTokenSection && (
                <div className="mt-2 p-4 bg-[var(--background)]/50 rounded-md border border-[var(--border-color)]">
                  <div className="mb-3">
                    <label className="block text-xs font-medium text-[var(--foreground)] mb-2">
                      {t.form?.selectPlatform || 'Select Platform'}
                    </label>
                    <div className="flex gap-2">
                      <button
                        type="button"
                        onClick={() => setSelectedPlatform('github')}
                        className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-md border transition-all ${selectedPlatform === 'github'
                          ? 'bg-[var(--accent-primary)]/10 border-[var(--accent-primary)] text-[var(--accent-primary)] shadow-sm'
                          : 'border-[var(--border-color)] text-[var(--foreground)] hover:bg-[var(--background)]'
                          }`}
                      >
                        <span className="text-sm">GitHub</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => setSelectedPlatform('gitlab')}
                        className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-md border transition-all ${selectedPlatform === 'gitlab'
                          ? 'bg-[var(--accent-primary)]/10 border-[var(--accent-primary)] text-[var(--accent-primary)] shadow-sm'
                          : 'border-[var(--border-color)] text-[var(--foreground)] hover:bg-[var(--background)]'
                          }`}
                      >
                        <span className="text-sm">GitLab</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => setSelectedPlatform('bitbucket')}
                        className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-md border transition-all ${selectedPlatform === 'bitbucket'
                          ? 'bg-[var(--accent-primary)]/10 border-[var(--accent-primary)] text-[var(--accent-primary)] shadow-sm'
                          : 'border-[var(--border-color)] text-[var(--foreground)] hover:bg-[var(--background)]'
                          }`}
                      >
                        <span className="text-sm">Bitbucket</span>
                      </button>
                    </div>
                  </div>
                  
                  <div>
                    <label htmlFor="access-token" className="block text-xs font-medium text-[var(--foreground)] mb-2">
                      {t.form?.personalAccessToken || 'Personal Access Token'}
                    </label>
                    <input
                      id="access-token"
                      type="password"
                      value={accessToken}
                      onChange={(e) => setAccessToken(e.target.value)}
                      placeholder={t.form?.tokenPlaceholder || 'Enter your access token'}
                      className="input-japanese block w-full px-3 py-2 rounded-md bg-transparent text-[var(--foreground)] focus:outline-none focus:border-[var(--accent-primary)] text-sm"
                    />
                    <div className="flex items-center mt-2 text-xs text-[var(--muted)]">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1 text-[var(--muted)]"
                        fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                          d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      {t.form?.tokenSecurityNote || 'Your token is stored locally and never sent to our servers.'}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Modal footer */}
          <div className="flex items-center justify-end gap-2 px-6 py-4 border-t border-[var(--border-color)]">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm font-medium rounded-md border border-[var(--border-color)]/50 text-[var(--muted)] bg-transparent hover:bg-[var(--background)] hover:text-[var(--foreground)] transition-colors"
            >
              {t.common?.cancel || 'Cancel'}
            </button>
            <button
              type="button"
              onClick={onSubmit}
              disabled={isSubmitting}
              className="px-4 py-2 text-sm font-medium rounded-md border border-transparent bg-[var(--accent-primary)]/90 text-white hover:bg-[var(--accent-primary)] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isSubmitting ? (t.common?.processing || 'Processing...') : (t.common?.generateWiki || 'Generate Wiki')}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
