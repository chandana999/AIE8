'use client'

import { useState, useRef, useEffect } from 'react'
import { Send, Bot, User, Settings, Sparkles, Upload, FileText, X, MessageSquare, Shield, AlertTriangle, Clock, Activity, Zap } from 'lucide-react'

interface Message {
  id: string
  content: string
  role: 'user' | 'assistant'
  timestamp: Date
}

interface LogStatus {
  log_loaded: boolean
  log_name: string | null
  chunks_count: number
  embeddings_available: boolean
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [developerMessage, setDeveloperMessage] = useState('You are a helpful AI assistant.')
  const [openaiApiKey, setOpenaiApiKey] = useState('')
  const [tavilyApiKey, setTavilyApiKey] = useState('')
  const [model, setModel] = useState('gpt-4o-mini')
  const [logType, setLogType] = useState('apache2')
  const [isLoading, setIsLoading] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [logStatus, setLogStatus] = useState<LogStatus>({ log_loaded: false, log_name: null, chunks_count: 0, embeddings_available: false })
  const [isUploading, setIsUploading] = useState(false)
  // Removed chatMode - only upload and sample traces available
  // Removed API key validation - keeping it simple like reference code
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const autoResizeTextArea = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`
    }
  }

  const getApiUrl = () => {
    return process.env.NODE_ENV === 'production' ? 'https://aie8.onrender.com' : 'http://localhost:8000'
  }

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    if (!openaiApiKey.trim()) {
      alert('Please enter your OpenAI API key in the settings first.')
      return
    }

    setIsUploading(true)

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('model', model)
      formData.append('api_key', openaiApiKey)
      formData.append('tavily_api_key', tavilyApiKey)

      const response = await fetch(`${getApiUrl()}/api/upload-log-file`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        // Try to get the error message from the backend
        let errorMessage = 'Failed to upload file'
        try {
          const errorData = await response.json()
          errorMessage = errorData.detail || errorMessage
        } catch {
          // If we can't parse the error, use the status text
          errorMessage = response.statusText || errorMessage
        }
        throw new Error(errorMessage)
      }

      const result = await response.json()
      
      setLogStatus({
        log_loaded: true,
        log_name: result.message.split("'")[1] || file.name,
        chunks_count: result.chunks_count,
        embeddings_available: true
      })
      // Set to log analysis mode (only mode available)
      
      // Display the analysis result if available
      if (result.analysis_result) {
        setMessages([{
          id: Date.now().toString(),
          content: result.analysis_result,
          role: 'assistant',
          timestamp: new Date()
        }])
      } else {
        setMessages([]) // Clear previous messages when switching to log analysis mode
      }
      
    } catch (error) {
      console.error('Upload error:', error)
      const errorMessage = error instanceof Error ? error.message : 'Failed to upload file'
      
      // Show specific error message to user
      setMessages([{
        id: Date.now().toString(),
        content: `❌ Upload Error: ${errorMessage}`,
        role: 'assistant',
        timestamp: new Date()
      }])
    } finally {
      setIsUploading(false)
    }
  }

  const handleQuickQuestion = (question: string) => {
    setInputMessage(question)
    setTimeout(() => {
      handleSendMessage()
    }, 100)
  }

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading || !openaiApiKey.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      role: 'user',
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInputMessage('')
    setIsLoading(true)

    try {
      // Only use log analysis - no general chat needed
      if (!logStatus.log_loaded) {
        throw new Error("Please upload a log file first to analyze")
      }
      
      const response = await fetch(`${getApiUrl()}/api/rag-chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          log_input: inputMessage,
          model: model,
          api_key: openaiApiKey,
          tavily_api_key: tavilyApiKey
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to get response')
      }

      // Handle streaming response (text/plain)
      const streamText = await response.text()
      
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: streamText || 'No response received',
        role: 'assistant',
        timestamp: new Date()
      }

      setMessages(prev => [...prev, assistantMessage])
      
    } catch (error) {
      console.error('Error:', error)
      const errorMessage = error instanceof Error ? error.message : 'An error occurred'
      
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        content: errorMessage,
        role: 'assistant',
        timestamp: new Date()
      }])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-gray-50 to-blue-50/30">
      <header className="bg-white shadow-sm border-b border-gray-100">
        <div className="max-w-6xl mx-auto px-6 py-5 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-gradient-to-r from-orange-600 to-red-600 rounded-xl flex items-center justify-center shadow-sm">
              <Shield className="w-7 h-7 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-semibold text-gray-900 tracking-tight">LogAnalyzer</h1>
              <p className="text-sm text-gray-500 font-medium">
                Apache Log Analysis
              </p>
            </div>
          </div>
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            aria-label="Settings"
          >
            <Settings className="w-6 h-6 text-gray-600" />
          </button>
        </div>
      </header>

      {showSettings && (
        <div className="bg-white border-b border-gray-200 p-4 animate-fade-in">
          <div className="max-w-6xl mx-auto space-y-4">
            {/* Log Analysis Mode - Only Mode Available */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Analysis Mode
              </label>
              <div className="flex gap-2">
                <button
                  className="px-4 py-2 rounded-lg text-sm font-medium bg-orange-600 text-white"
                >
                  <FileText className="w-4 h-4 inline mr-2" />
                  Log Analysis
                </button>
              </div>
              <p className="text-xs text-gray-500 mt-1">Focused on log analysis with incident knowledge base</p>
            </div>

            {/* Log Type Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Log Type
              </label>
              <select
                value={logType}
                onChange={(e) => setLogType(e.target.value)}
                className="input-field"
                aria-label="Log type"
              >
                <option value="apache2">Apache2 Web Server</option>
                <option value="apache">Apache Web Server</option>
                <option value="nginx">Nginx Web Server</option>
              </select>
            </div>

            {/* API Keys */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2" htmlFor="openaiApiKey">
                  OpenAI API Key
                </label>
                <input
                  id="openaiApiKey"
                  type="password"
                  value={openaiApiKey}
                  onChange={(e) => setOpenaiApiKey(e.target.value)}
                  placeholder="sk-..."
                  className="input-field"
                  aria-label="OpenAI API Key"
                  autoComplete="off"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2" htmlFor="tavilyApiKey">
                  Tavily API Key
                </label>
                <input
                  id="tavilyApiKey"
                  type="password"
                  value={tavilyApiKey}
                  onChange={(e) => setTavilyApiKey(e.target.value)}
                  placeholder="tvly-..."
                  className="input-field"
                  aria-label="Tavily API Key"
                  autoComplete="off"
                />
              </div>
            </div>
            
            {/* Removed system message - not needed for log analysis */}
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2" htmlFor="modelSelect">
                Model
              </label>
              <select
                id="modelSelect"
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="input-field"
                aria-label="Model selection"
              >
                <option value="gpt-4o-mini">GPT-4o Mini</option>
                <option value="gpt-4o">GPT-4o</option>
                <option value="gpt-4-turbo">GPT-4 Turbo</option>
              </select>
            </div>
          </div>
        </div>
      )}

      <div className="flex-1 flex flex-col max-w-6xl mx-auto w-full">
        {/* File Upload Section */}
        <div className="bg-white border-b border-gray-200 p-6">
          <div className="flex items-center justify-center">
            <div className="text-center">
              <input
                ref={fileInputRef}
                type="file"
                accept=".log,.txt"
                onChange={handleFileUpload}
                className="hidden"
                id="file-upload"
                aria-label="Upload log file"
              />
              <label
                htmlFor="file-upload"
                className="cursor-pointer inline-flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-orange-600 to-red-600 text-white rounded-2xl font-medium shadow-lg hover:shadow-xl transition-all duration-200 transform hover:scale-105"
              >
                {isUploading ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Upload className="w-5 h-5" />
                    Upload Log File
                  </>
                )}
              </label>
              <p className="text-sm text-gray-500 mt-2">
                {logStatus.log_loaded 
                  ? `✅ ${logStatus.log_name} loaded (${logStatus.chunks_count} chunks)`
                  : 'Supported formats: .log, .txt'
                }
              </p>
            </div>
          </div>
        </div>

        {/* Chat Interface */}
        <div className="flex-1 flex flex-col min-h-0">
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {messages.length === 0 ? (
              <div className="text-center mt-16 select-none">
                <Shield className="w-16 h-16 mx-auto mb-8 text-orange-200" />
                <h3 className="text-3xl font-semibold mb-6 text-gray-800 tracking-tight">Welcome to LogAnalyzer</h3>
                <div className="max-w-2xl mx-auto">
                  <p className="text-lg mb-4 text-gray-600 font-medium">Upload your Apache logs for instant AI-powered analysis</p>
                  <p className="text-base text-gray-500">Get detailed incident analysis with root cause identification and remediation steps</p>
                </div>
                
                {/* Log Type Suggestions */}
                <div className="mt-12 p-6 bg-blue-50/80 rounded-2xl border border-blue-100 shadow-sm">
                  <h4 className="text-base font-semibold text-blue-900 mb-4">📋 Supported Log Types - Upload These for Analysis:</h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <div className="p-3 bg-white rounded-xl border border-blue-200">
                      <div className="flex items-center mb-2">
                        <AlertTriangle className="w-4 h-4 text-red-500 mr-2" />
                        <span className="font-medium text-sm">500 Internal Server Errors</span>
                      </div>
                      <p className="text-xs text-gray-600">PHP fatal errors, memory issues, database connection failures</p>
                    </div>
                    <div className="p-3 bg-white rounded-xl border border-blue-200">
                      <div className="flex items-center mb-2">
                        <Clock className="w-4 h-4 text-orange-500 mr-2" />
                        <span className="font-medium text-sm">504 Gateway Timeout</span>
                      </div>
                      <p className="text-xs text-gray-600">Upstream server timeouts, proxy issues, slow backend responses</p>
                    </div>
                    <div className="p-3 bg-white rounded-xl border border-blue-200">
                      <div className="flex items-center mb-2">
                        <Shield className="w-4 h-4 text-red-600 mr-2" />
                        <span className="font-medium text-sm">403 Forbidden Errors</span>
                      </div>
                      <p className="text-xs text-gray-600">Access control violations, permission issues, blocked requests</p>
                    </div>
                    <div className="p-3 bg-white rounded-xl border border-blue-200">
                      <div className="flex items-center mb-2">
                        <FileText className="w-4 h-4 text-blue-500 mr-2" />
                        <span className="font-medium text-sm">404 Not Found</span>
                      </div>
                      <p className="text-xs text-gray-600">Missing files, broken links, configuration issues</p>
                    </div>
                    <div className="p-3 bg-white rounded-xl border border-blue-200">
                      <div className="flex items-center mb-2">
                        <Zap className="w-4 h-4 text-purple-500 mr-2" />
                        <span className="font-medium text-sm">502 Bad Gateway</span>
                      </div>
                      <p className="text-xs text-gray-600">Proxy errors, backend server issues, load balancer problems</p>
                    </div>
                    <div className="p-3 bg-white rounded-xl border border-blue-200">
                      <div className="flex items-center mb-2">
                        <Activity className="w-4 h-4 text-green-500 mr-2" />
                        <span className="font-medium text-sm">Access Logs</span>
                      </div>
                      <p className="text-xs text-gray-600">Traffic patterns, suspicious activity, performance issues</p>
                    </div>
                  </div>
                  <p className="text-xs text-blue-700 mt-3">
                    💡 <strong>Tip:</strong> Upload any Apache error or access log file to get instant analysis with error types, root causes, remediation steps, and prevention recommendations.
                  </p>
                </div>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] px-5 py-4 rounded-2xl shadow-sm ${
                      message.role === 'user'
                        ? 'bg-orange-600 text-white ml-auto'
                        : 'bg-gray-50 text-gray-800 border border-gray-100'
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      {message.role === 'assistant' && (
                        <Bot className="w-5 h-5 text-gray-500 mt-1 flex-shrink-0" />
                      )}
                      <div className="flex-1 break-words text-base leading-relaxed">
                        <div className="bg-gradient-to-r from-gray-50 to-blue-50 p-6 rounded-xl border border-gray-200 shadow-sm">
                          <div className="space-y-4">
                            {message.content.split('\n').map((line, index) => {
                              // Handle main analysis header
                              if (line.includes('=== LogAnalysisRAG Analysis ===')) {
                                return (
                                  <div key={index} className="bg-gradient-to-r from-orange-500 to-red-500 text-white px-4 py-2 rounded-lg font-bold text-center mb-4">
                                    🤖 AI Log Analysis Results
                                  </div>
                                )
                              }
                              // Handle section headers
                              else if (line.startsWith('###')) {
                                const cleanLine = line.replace(/^###\s*/, '').replace(/\*/g, '')
                                return (
                                  <div key={index} className="font-bold text-lg text-gray-800 border-b border-gray-300 pb-2">
                                    {cleanLine}
                                  </div>
                                )
                              }
                              // Handle bullet points
                              else if (line.startsWith('-')) {
                                const cleanLine = line.replace(/^-\s*/, '').replace(/\*/g, '')
                                return (
                                  <div key={index} className="ml-4 flex items-start">
                                    <span className="text-orange-500 mr-2 mt-1">•</span>
                                    <span className="text-gray-700">{cleanLine}</span>
                                  </div>
                                )
                              }
                              // Handle regular text
                              else if (line.trim()) {
                                const cleanLine = line.replace(/\*/g, '')
                                return (
                                  <div key={index} className="text-gray-700 leading-relaxed">
                                    {cleanLine}
                                  </div>
                                )
                              }
                              // Handle empty lines
                              return <div key={index} className="h-2"></div>
                            })}
                          </div>
                        </div>
                        <p className="text-xs opacity-60 mt-3 select-none font-medium">
                          {message.timestamp.toLocaleTimeString()}
                        </p>
                      </div>
                      {message.role === 'user' && (
                        <User className="w-5 h-5 text-white mt-1 flex-shrink-0" />
                      )}
                    </div>
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className="flex justify-start">
                <div className="chat-message max-w-[80%] bg-gray-100 text-gray-900 px-4 py-2 rounded-lg">
                  <div className="flex items-center gap-2">
                    <Bot className="w-5 h-5 text-gray-500" />
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                      <div
                        className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                        style={{ animationDelay: '0.1s' }}
                      ></div>
                      <div
                        className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                        style={{ animationDelay: '0.2s' }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {/* Analysis Complete Message */}
            {logStatus.log_loaded && messages.length > 0 && (
              <div className="mt-4 p-3 bg-green-50 rounded-lg border border-green-200">
                <div className="flex items-center">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                  <span className="text-xs font-medium text-green-800">
                    ✅ Log analysis complete! Upload another log file for more analysis.
                  </span>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          <div className="border-t border-gray-200 p-6">
            {/* Sample Log Traces - Compact (only when log is loaded) */}
            {logStatus.log_loaded && (
              <div className="mb-3">
                <h4 className="text-sm font-semibold text-blue-900 mb-2">📋 Sample Traces:</h4>
                <div className="flex gap-2">
                <button
                  onClick={() => handleQuickQuestion("2024-01-15T09:59:58.990Z [INFO] [trace_id:req-000-init] [request_id:init-001] [ELB:frontend-sg] [AZ:us-east-1a] [EC2:i-0123456789abcd001] System starting up - Initializing web tier health checks\n2024-01-15T10:00:00.111Z [WARN] [trace_id:req-001-certcheck] [request_id:req-001] [ELB:frontend-sg] [AZ:us-east-1a] SSL certificate will expire in 1 day - subject: CN=api.example.com\n2024-01-15T10:00:01.002Z [ERROR] [trace_id:req-002-certfail] [request_id:req-002] [ELB:frontend-sg] [AZ:us-east-1a] GET /api/auth - SSL Certificate Expired - 5000ms - Certificate expired on 2024-01-15\n2024-01-15T10:00:01.123Z [ERROR] [trace_id:req-003-timeout] [request_id:req-003] [ELB:frontend-sg] [AZ:us-east-1b] GET /api/orders - 504 Gateway Timeout - 30000ms - Backend service unavailable due to SSL handshake failure\n2024-01-15T10:00:01.267Z [WARN] [trace_id:req-004-badgateway] [request_id:req-004] [ELB:frontend-sg] [AZ:us-east-1a] POST /api/cart - 502 Bad Gateway - 8000ms - Connection refused to backend API node\n2024-01-15T10:00:01.445Z [ERROR] [trace_id:req-005-overload] [request_id:req-005] [ELB:frontend-sg] [AZ:us-east-1a] GET /api/products - 503 Service Unavailable - 4000ms - Connection pool exhausted from retry storm\n2024-01-15T10:00:01.556Z [ERROR] [trace_id:req-006-internal] [request_id:req-006] [ELB:frontend-sg] [AZ:us-east-1b] GET /api/users - 500 Internal Server Error - 2000ms - PHP Fatal error: Call to undefined function handle_retry()\n2024-01-15T10:00:01.667Z [ERROR] [trace_id:req-007-forbidden] [request_id:req-007] [ELB:frontend-sg] [AZ:us-east-1a] POST /api/admin/users - 403 Forbidden - 15ms - Insufficient permissions after auth fallback\n2024-01-15T10:00:01.778Z [ERROR] [trace_id:req-008-forbidden] [request_id:req-008] [ELB:frontend-sg] [AZ:us-east-1b] GET /api/admin/logs - 403 Forbidden - 10ms - Access denied due to invalid session token\n2024-01-15T10:00:01.889Z [ERROR] [trace_id:req-009-recovery] [request_id:req-009] [ELB:frontend-sg] [AZ:us-east-1a] POST /api/recovery/trigger - 200 OK - 300ms - Restart initiated for backend API node\n2024-01-15T10:00:02.000Z [INFO] [trace_id:req-010-restart] [request_id:req-010] [EC2:i-0123456789abcd002] Backend API restarting - applying renewed SSL certificate\n2024-01-15T10:00:02.334Z [INFO] [trace_id:req-011-recover] [request_id:req-011] [ELB:frontend-sg] Health checks passed - Backend API node restored successfully")}
                  className="text-left px-3 py-2 bg-white rounded-lg border border-blue-200 hover:bg-blue-50 transition-all duration-200"
                >
                  <div className="flex items-center">
                    <AlertTriangle className="w-3 h-3 text-red-500 mr-1" />
                    <span className="text-xs font-medium">SSL Cascade</span>
                    <span className="ml-1 text-xs bg-red-100 text-red-800 px-1 py-0.5 rounded">High</span>
                  </div>
                </button>
                
                <button
                  onClick={() => handleQuickQuestion("2024-03-20T14:00:00.001Z [INFO] [trace_id:req-000-init] [request_id:init-001] [ELB:frontend-sg] [AZ:us-west-2a] [EC2:i-0abcd123456789ef0] System startup - Web service health checks green\n2024-03-20T14:00:02.144Z [WARN] [trace_id:req-002-db-latency] [request_id:req-002] [RDS:db-primary] [AZ:us-west-2a] Detected increased write latency - possible lock contention\n2024-03-20T14:00:03.287Z [ERROR] [trace_id:req-003-db-deadlock] [request_id:req-003] [RDS:db-primary] [AZ:us-west-2a] Transaction deadlock detected - rolling back session id 2187\n2024-03-20T14:00:03.411Z [ERROR] [trace_id:req-004-api-fail] [request_id:req-004] [EC2:i-0abcd123456789ef1] POST /api/orders - 504 Gateway Timeout - 30000ms - Query timeout waiting for DB response\n2024-03-20T14:00:03.612Z [ERROR] [trace_id:req-005-api-retry] [request_id:req-005] [EC2:i-0abcd123456789ef1] POST /api/orders - 502 Bad Gateway - 12000ms - Connection reset during retry\n2024-03-20T14:00:03.879Z [ERROR] [trace_id:req-006-checkout] [request_id:req-006] [ELB:frontend-sg] [AZ:us-west-2b] POST /api/checkout - 504 Gateway Timeout - 30000ms - Backend not responding\n2024-03-20T14:00:04.102Z [ERROR] [trace_id:req-007-backend-overload] [request_id:req-007] [EC2:i-0abcd123456789ef2] GET /api/inventory - 503 Service Unavailable - 6000ms - Connection pool saturated\n2024-03-20T14:00:04.334Z [ERROR] [trace_id:req-008-internal] [request_id:req-008] [EC2:i-0abcd123456789ef2] GET /api/inventory - 500 Internal Server Error - 3000ms - Python RuntimeError: max recursion depth exceeded\n2024-03-20T14:00:04.556Z [WARN] [trace_id:req-009-retry-storm] [request_id:req-009] [ELB:frontend-sg] [AZ:us-west-2a] Warning - High retry rate detected (500 req/s) - throttling initiated\n2024-03-20T14:00:04.778Z [INFO] [trace_id:req-010-recovery] [request_id:req-010] [RDS:db-primary] [AZ:us-west-2a] Database lock released - replication sync restored\n2024-03-20T14:00:05.001Z [INFO] [trace_id:req-011-stable] [request_id:req-011] [ELB:frontend-sg] [AZ:us-west-2b] System health check: all services operational after DB recovery")}
                  className="text-left px-3 py-2 bg-white rounded-lg border border-blue-200 hover:bg-blue-50 transition-all duration-200"
                >
                  <div className="flex items-center">
                    <Activity className="w-3 h-3 text-orange-500 mr-1" />
                    <span className="text-xs font-medium">DB Cascade</span>
                    <span className="ml-1 text-xs bg-orange-100 text-orange-800 px-1 py-0.5 rounded">Medium</span>
                  </div>
                </button>
              </div>
            </div>
            )}

            {/* Upload Option */}
            <div className="text-center">
              <div className="flex items-center justify-center gap-3 mb-4">
                <Upload className="w-5 h-5 text-orange-600" />
                <span className="text-lg font-medium text-gray-700">Upload Your Own Log File</span>
              </div>
              <p className="text-sm text-gray-500 mb-4">
                {!logStatus.log_loaded 
                  ? "Upload a log file above to get instant AI-powered analysis"
                  : "Analysis complete! Upload another log file for more analysis"
                }
              </p>
              {!openaiApiKey.trim() && (
                <p className="text-sm text-red-500" role="alert">
                  Please enter your OpenAI API key in the settings to start analyzing logs.
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
