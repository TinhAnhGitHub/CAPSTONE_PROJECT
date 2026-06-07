import { useGoogleLogin } from '@react-oauth/google'
import React, { useEffect } from 'react'
import api from '@/api/api';
import { useNavigate } from 'react-router-dom';
import { useStore } from '@/stores/user';

export default function Login() {
    const navigate = useNavigate();
    const { login } = useStore.getState();
    const googleLogin = useGoogleLogin({
        flow: 'auth-code',
        onSuccess: async (codeResponse) => {
            const { data } = await api.post(
                'api/user/login/google', {
                code: codeResponse.code,
            });
            const userInfo = data.user;
            login(userInfo, data.access_token);
            navigate('/');
        },
        onError: errorResponse => console.error(errorResponse),
    });

    useEffect(() => {
        if (useStore.getState().token) {
            navigate('/');
        }
    }, []);

    return (
        <div className="relative h-screen w-screen flex items-center justify-center overflow-hidden bg-background">

            {/* ── Square grid background ── */}
            <div
                className="absolute inset-0 pointer-events-none"
                style={{
                    backgroundImage: `
                        linear-gradient(rgba(0,173,181,0.08) 1px, transparent 1px),
                        linear-gradient(90deg, rgba(0,173,181,0.08) 1px, transparent 1px)
                    `,
                    backgroundSize: '48px 48px',
                }}
            />

            {/* ── Big outline video icon centered in BG ── */}
            {/* <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="rgba(0,173,181,0.07)"
                    strokeWidth="0.3"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="w-[70vmin] h-[70vmin]"
                >
                    <rect x="2" y="3" width="20" height="14" rx="2" />
                    <path d="M8 3v14M16 3v14M2 8h4M18 8h4M2 12h4M18 12h4" />
                    <circle cx="17.5" cy="19.5" r="2.5" />
                    <path d="M20 22l1.5 1.5" />
                </svg>
            </div> */}

            {/* ── Glass card ── */}
            <div
                className="relative z-10 w-full max-w-md mx-4"
                style={{ animation: 'card-enter 0.6s cubic-bezier(0.22,1,0.36,1) both' }}
            >
                <div
                    className="rounded-3xl border border-white/10 p-10 flex flex-col items-center gap-8"
                    style={{
                        background: 'rgba(48, 48, 48, 0.65)',
                        backdropFilter: 'blur(20px)',
                        WebkitBackdropFilter: 'blur(20px)',
                        boxShadow: '0 8px 64px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.07)',
                    }}
                >
                    {/* Title */}
                    <div className="text-center">
                        <h1
                            className="text-3xl font-bold tracking-tight"
                            style={{
                                background: '#00ADB5',
                                WebkitBackgroundClip: 'text',
                                WebkitTextFillColor: 'transparent',
                                backgroundClip: 'text',
                            }}
                        >
                            VideoDeepSearch
                        </h1>
                        <p className="mt-1.5 text-sm text-text-dim tracking-wide">
                            Semantic Video Retrieval System
                        </p>
                    </div>

                    {/* Divider */}
                    <div className="w-full flex items-center gap-3">
                        <div className="flex-1 h-px bg-white/10" />
                        <span className="text-xs text-text-dim uppercase tracking-widest">Sign in to continue</span>
                        <div className="flex-1 h-px bg-white/10" />
                    </div>

                    {/* Google button */}
                    <button
                        onClick={() => googleLogin()}
                        className="group relative flex items-center gap-3 w-full px-5 py-3.5 rounded-xl font-medium text-sm transition-all duration-200 cursor-pointer overflow-hidden"
                        style={{
                            background: 'rgba(255,255,255,0.06)',
                            border: '1px solid rgba(255,255,255,0.12)',
                            color: '#EEEEEE',
                        }}
                    >
                        {/* Hover shimmer */}
                        <span
                            className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"
                            style={{ background: 'linear-gradient(135deg, rgba(0,173,181,0.12), rgba(0,173,181,0.04))' }}
                        />
                        <span
                            className="absolute inset-0 opacity-0 group-active:opacity-100 transition-opacity duration-100 pointer-events-none"
                            style={{ background: 'rgba(0,173,181,0.15)' }}
                        />

                        <svg className="w-5 h-5 shrink-0 relative z-10" viewBox="0 0 24 24">
                            <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
                            <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                            <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
                            <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
                        </svg>

                        <span className="relative z-10 flex-1 text-center group-hover:text-white transition-colors">
                            Continue with Google
                        </span>

                        <svg className="w-4 h-4 relative z-10 text-text-dim group-hover:text-accent group-hover:translate-x-0.5 transition-all duration-200" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M7.21 14.77a.75.75 0 01.02-1.06L11.168 10 7.23 6.29a.75.75 0 111.04-1.08l4.5 4.25a.75.75 0 010 1.08l-4.5 4.25a.75.75 0 01-1.06-.02z" clipRule="evenodd" />
                        </svg>
                    </button>

                    {/* Footer */}
                    <p className="text-center text-xs text-text-dim leading-relaxed">
                        Capstone Project · Ho Chi Minh City University of Technology
                    </p>
                </div>
            </div>

            <style>{`
                @keyframes card-enter {
                    from { opacity: 0; transform: translateY(20px) scale(0.97); }
                    to   { opacity: 1; transform: translateY(0) scale(1); }
                }
            `}</style>
        </div>
    )
}
