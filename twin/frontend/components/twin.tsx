'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, User, MessageCircle, ArrowRight } from 'lucide-react';

interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
}

export default function Twin() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [sessionId, setSessionId] = useState<string>('');
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const sendMessage = async () => {
        if (!input.trim() || isLoading) return;

        const userMessage: Message = {
            id: Date.now().toString(),
            role: 'user',
            content: input,
            timestamp: new Date(),
        };

        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: input,
                    session_id: sessionId || undefined,
                }),
            });

            if (!response.ok) throw new Error('Failed to send message');

            const data = await response.json();

            if (!sessionId) {
                setSessionId(data.session_id);
            }

            const assistantMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: data.response,
                timestamp: new Date(),
            };

            setMessages(prev => [...prev, assistantMessage]);
        } catch (error) {
            console.error('Error:', error);
            const errorMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: 'Sorry, I encountered an error. Please try again.',
                timestamp: new Date(),
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    const suggestedQuestions = [
        "What's your experience with AI/ML?",
        "What technologies do you specialize in?",
    ];

    return (
        <div className="flex flex-col h-full bg-gradient-to-br from-white via-blue-50 to-indigo-50 rounded-3xl shadow-2xl overflow-hidden border border-indigo-100/50">
            {/* Header */}
            <div className="bg-gradient-to-r from-indigo-600 via-blue-600 to-cyan-500 text-white p-8 border-b border-indigo-400/30 relative overflow-hidden">
                {/* Decorative background elements */}
                <div className="absolute inset-0 opacity-10">
                    <div className="absolute top-0 right-0 w-96 h-96 bg-white rounded-full blur-3xl transform translate-x-1/3 -translate-y-1/4" />
                </div>

                <div className="relative z-10 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <div className="p-3 bg-white/20 rounded-2xl backdrop-blur-md border border-white/20 shadow-lg">
                            <Sparkles className="w-7 h-7" />
                        </div>
                        <div>
                            <h2 className="text-3xl font-bold bg-gradient-to-r from-white to-indigo-100 bg-clip-text text-transparent">
                                My Digital Career Twin
                            </h2>
                            <p className="text-sm text-indigo-100 mt-1 font-medium">Powered by AWS Bedrock • RAG-Enhanced</p>
                        </div>
                    </div>
                    <MessageCircle className="w-6 h-6 text-indigo-200 opacity-60 hidden sm:block" />
                </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-8 space-y-6">
                {messages.length === 0 && (
                    <div className="flex flex-col items-center justify-center h-full text-center py-12">
                        <div className="mb-8 animate-bounce" style={{ animationDuration: '3s' }}>
                            <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-br from-indigo-100 via-blue-100 to-cyan-100 shadow-xl">
                                <Sparkles className="w-10 h-10 text-indigo-600" />
                            </div>
                        </div>
                        <h3 className="text-2xl font-bold text-gray-900 mb-3">Career Twin - Get to Know Priya</h3>
                        <p className="text-gray-600 max-w-md mb-8 leading-relaxed">
                            Ask me anything about Priya's experience, skills, projects, or career journey. I have all her details and can answer your questions instantly.
                        </p>

                        {/* Suggested Questions */}
                        <div className="w-full max-w-2xl space-y-3">
                            <p className="text-sm text-gray-500 font-semibold uppercase tracking-wider">Suggested Questions</p>
                            {suggestedQuestions.map((question, index) => (
                                <button
                                    key={index}
                                    onClick={() => {
                                        setInput(question);
                                    }}
                                    className="w-full text-left p-4 rounded-2xl bg-white border-2 border-indigo-100 hover:border-indigo-400 hover:bg-indigo-50 transition-all duration-300 group cursor-pointer shadow-sm hover:shadow-md"
                                >
                                    <div className="flex items-center justify-between">
                                        <span className="text-gray-700 font-medium group-hover:text-indigo-600 transition-colors">{question}</span>
                                        <ArrowRight className="w-4 h-4 text-indigo-400 group-hover:translate-x-1 transition-transform opacity-0 group-hover:opacity-100" />
                                    </div>
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                {messages.map((message) => (
                    <div
                        key={message.id}
                        className={`flex gap-4 animate-in fade-in slide-in-from-bottom-2 ${
                            message.role === 'user' ? 'justify-end' : 'justify-start'
                        }`}
                    >
                        {message.role === 'assistant' && (
                            <div className="flex-shrink-0 mt-1">
                                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-indigo-500 to-blue-500 flex items-center justify-center shadow-lg border border-indigo-300">
                                    <Sparkles className="w-5 h-5 text-white" />
                                </div>
                            </div>
                        )}

                        <div
                            className={`max-w-[70%] rounded-3xl px-6 py-4 ${
                                message.role === 'user'
                                    ? 'bg-gradient-to-br from-indigo-600 to-blue-600 text-white shadow-lg border border-indigo-500'
                                    : 'bg-white text-gray-800 shadow-lg border border-indigo-100 backdrop-blur-sm'
                            }`}
                        >
                            <p className="text-sm leading-relaxed whitespace-pre-wrap font-medium">
                                {message.content}
                            </p>
                            <p
                                className={`text-xs mt-3 font-semibold ${
                                    message.role === 'user' ? 'text-indigo-100' : 'text-gray-400'
                                }`}
                            >
                                {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                            </p>
                        </div>

                        {message.role === 'user' && (
                            <div className="flex-shrink-0 mt-1">
                                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-gray-700 to-gray-900 flex items-center justify-center shadow-lg border border-gray-600">
                                    <User className="w-5 h-5 text-white" />
                                </div>
                            </div>
                        )}
                    </div>
                ))}

                {isLoading && (
                    <div className="flex gap-4 justify-start">
                        <div className="flex-shrink-0 mt-1">
                            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-indigo-500 to-blue-500 flex items-center justify-center shadow-lg border border-indigo-300">
                                <Sparkles className="w-5 h-5 text-white" />
                            </div>
                        </div>
                        <div className="bg-white rounded-3xl p-5 shadow-lg border border-indigo-100 flex space-x-3">
                            <div className="w-3 h-3 bg-indigo-500 rounded-full animate-bounce" style={{ animationDelay: '0s' }} />
                            <div className="w-3 h-3 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.15s' }} />
                            <div className="w-3 h-3 bg-cyan-500 rounded-full animate-bounce" style={{ animationDelay: '0.3s' }} />
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="border-t border-indigo-200/50 p-6 bg-white/60 backdrop-blur-xl">
                <div className="flex gap-3">
                    <div className="relative flex-1">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={handleKeyPress}
                            placeholder="Ask about Priya's experience, skills, or projects..."
                            className="w-full px-6 py-4 border-2 border-indigo-200 rounded-2xl focus:outline-none focus:border-indigo-500 focus:ring-3 focus:ring-indigo-200/50 text-gray-800 font-medium placeholder-gray-500 transition-all shadow-sm focus:shadow-md"
                            disabled={isLoading}
                        />
                    </div>
                    <button
                        onClick={sendMessage}
                        disabled={!input.trim() || isLoading}
                        className="px-7 py-4 bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-700 hover:to-blue-700 text-white rounded-2xl font-semibold shadow-xl hover:shadow-2xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 group"
                    >
                        <Send className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                        <span className="hidden sm:inline">Send</span>
                    </button>
                </div>
            </div>
        </div>
    );
}