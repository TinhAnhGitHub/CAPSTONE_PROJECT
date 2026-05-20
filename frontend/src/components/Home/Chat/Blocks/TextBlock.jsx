import { useState, useRef } from "react";
import { createPortal } from "react-dom";
import clsx from "clsx";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { ClipboardDocumentIcon, CheckIcon } from "@heroicons/react/24/outline";
import toast from "react-hot-toast";
import { useStore as useChatStore } from "@/stores/chat";
import { useVideoModalStore } from "@/stores/videoModal";

// Matches a 24-char hex ObjectId whether bare, in "quotes", or `backtick` spans.
// Capture group 1 = the raw ID (no surrounding punctuation).
const OBJECT_ID_RE = /["'`]?([0-9a-f]{24})["'`]?/g;

// not using
// Matches `hh:mm:ss.mmm - hh:mm:ss.mmm | video_id` or `hh:mm:ss.mmm | video_id`
// inside a backtick-wrapped inline-code node (content only, no backticks).
// Groups: 1=start, 2=end (optional), 3=videoId
const TIMESTAMP_RE = /^(\d{2}:\d{2}:\d{2}\.\d{3})(?:\s*-\s*(\d{2}:\d{2}:\d{2}\.\d{3}))?\s*\|\s*([0-9a-f]{24})$/;

// not using
/** Convert hh:mm:ss.mmm string to total seconds (float). */
function tsToSeconds(ts) {
    const [hh, mm, rest] = ts.split(':');
    const [ss, ms] = rest.split('.');
    return parseInt(hh, 10) * 3600 + parseInt(mm, 10) * 60 + parseInt(ss, 10) + parseInt(ms, 10) / 1000;
}

/**
 * Clickable chip for a video ObjectId. Looks up the video in workspaceVideos
 * and calls useVideoModalStore.open(video) directly — no <a> tag involved.
 */
const MINIO_BASE = import.meta.env.VITE_PRIMARY_URL + "/media/videos";
    
function VideoIdChip({ videoId }) {
    const workspaceVideos = useChatStore((s) => s.workspaceVideos);
    const open = useVideoModalStore((s) => s.open);
    const [showPreview, setShowPreview] = useState(false);
    const [tooltipPos, setTooltipPos] = useState({ top: 0, left: 0 });
    const hoverTimerRef = useRef(null);
    const buttonRef = useRef(null);

    // Try to find a richer record (name, thumbnail) in workspace; fall back to ID-only
    const workspaceVideo = workspaceVideos.find((v) => v._id === videoId || v.id === videoId);
    const videoObj = {
        url: `${MINIO_BASE}/${videoId}.mp4`,
        name: workspaceVideo?.name ?? videoId,
        thumbnail: workspaceVideo?.thumbnail ?? null,
    };

    const mousePosRef = useRef({ x: 0, y: 0 });

    const showTooltip = () => {
        const { x, y } = mousePosRef.current;
        const TOOLTIP_W = 224; // w-56 = 14rem = 224px
        // Center the tooltip horizontally on the cursor, clamped to viewport edges
        const left = Math.max(8, Math.min(x - TOOLTIP_W / 2, window.innerWidth - TOOLTIP_W - 8));
        const top = y; // used to compute 'bottom' in render
        setTooltipPos({ top, left });
        setShowPreview(true);
    };

    // ── Desktop: show on hover (400ms delay) ──
    const handleMouseMove = (e) => {
        mousePosRef.current = { x: e.clientX, y: e.clientY };
    };
    const handleMouseEnter = (e) => {
        mousePosRef.current = { x: e.clientX, y: e.clientY };
        hoverTimerRef.current = setTimeout(showTooltip, 400);
    };
    const handleMouseLeave = () => {
        clearTimeout(hoverTimerRef.current);
        setShowPreview(false);
    };

    // ── Mobile: show on long press (600ms) ──
    const longPressRef = useRef(null);
    const autoDismissRef = useRef(null);

    const handleTouchStart = () => {
        longPressRef.current = setTimeout(() => {
            showTooltip();
            // auto-dismiss after 2.5s
            autoDismissRef.current = setTimeout(() => setShowPreview(false), 2500);
        }, 600);
    };
    const handleTouchEnd = () => {
        clearTimeout(longPressRef.current);
    };
    const handleTouchMove = () => {
        clearTimeout(longPressRef.current);
        clearTimeout(autoDismissRef.current);
        setShowPreview(false);
    };

    return (
        <span
            className="relative inline-flex"
            onMouseEnter={handleMouseEnter}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
            onTouchStart={handleTouchStart}
            onTouchEnd={handleTouchEnd}
            onTouchMove={handleTouchMove}
        >
            <button
                ref={buttonRef}
                onClick={() => open(videoObj)}
                title={videoObj.name}
                className={clsx(
                    'inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-mono align-baseline',
                    'border transition-all cursor-pointer',
                    'bg-accent/15 border-accent/40 text-accent hover:bg-accent/30 hover:border-accent active:bg-accent/40 active:scale-95'
                )}
            >
                <svg className="w-3 h-3 shrink-0" viewBox="0 0 20 20" fill="currentColor">
                    <path d="M2 6a2 2 0 012-2h1v2H4a1 1 0 00-1 1v6a1 1 0 001 1h1v2H4a2 2 0 01-2-2V6zM15 4h1a2 2 0 012 2v8a2 2 0 01-2 2h-1v-2h1a1 1 0 001-1V7a1 1 0 00-1-1h-1V4zM6 4h8v2H6V4zm0 10h8v2H6v-2zm0-5h8v2H6V9z" />
                </svg>
                {videoObj.name}
            </button>

            {/* Hover / long-press video preview tooltip — portalled to body to escape overflow clipping */}
            {showPreview && createPortal(
                <div
                    className="fixed z-[9999] w-56 rounded-xl overflow-hidden pointer-events-none"
                    style={{
                        bottom: `${window.innerHeight - tooltipPos.top}px`,
                        left: tooltipPos.left,
                        background: 'rgba(48,48,48,0.97)',
                        border: '1px solid rgba(255,255,255,0.1)',
                        boxShadow: '0 8px 32px rgba(0,0,0,0.6), 0 0 0 1px rgba(0,173,181,0.2)',
                        animation: 'chip-preview-in 0.15s ease-out both',
                    }}
                >
                    {/* Video / thumbnail */}
                    <div className="relative w-full aspect-video bg-surface-light">
                        <video
                            src={videoObj.url}
                            poster={videoObj.thumbnail ?? undefined}
                            className="w-full h-full object-cover"
                            autoPlay
                            muted
                            loop
                            playsInline
                        />
                        {/* Accent top bar */}
                        <div className="absolute top-0 inset-x-0 h-px bg-accent/60" />
                    </div>
                    <div className="px-3 py-2">
                        <p className="text-xs font-medium text-text truncate">{videoObj.name}</p>
                        <p className="text-[10px] text-accent/80 mt-0.5">Click to open player</p>
                    </div>
                </div>,
                document.body
            )}
        </span>
    );
}

/**
 * Clickable chip for a timestamp + video_id.
 * Matches: `hh:mm:ss.mmm - hh:mm:ss.mmm | video_id` or `hh:mm:ss.mmm | video_id`
 * Clicking opens the video modal seeked to the start time.
 */
function TimestampChip({ startTs, endTs, videoId }) {
    const workspaceVideos = useChatStore((s) => s.workspaceVideos);
    const open = useVideoModalStore((s) => s.open);
    const [showPreview, setShowPreview] = useState(false);
    const [tooltipPos, setTooltipPos] = useState({ top: 0, left: 0 });
    const hoverTimerRef = useRef(null);
    const mousePosRef = useRef({ x: 0, y: 0 });

    const workspaceVideo = workspaceVideos.find((v) => v._id === videoId || v.id === videoId);
    const videoObj = {
        url: `${MINIO_BASE}/${videoId}.mp4`,
        name: workspaceVideo?.name ?? videoId,
        thumbnail: workspaceVideo?.thumbnail ?? null,
    };

    const label = endTs ? `${startTs} – ${endTs}` : startTs;
    const startSeconds = tsToSeconds(startTs);

    const showTooltip = () => {
        const { x, y } = mousePosRef.current;
        const TOOLTIP_W = 224;
        const left = Math.max(8, Math.min(x - TOOLTIP_W / 2, window.innerWidth - TOOLTIP_W - 8));
        setTooltipPos({ top: y, left });
        setShowPreview(true);
    };

    const handleMouseMove = (e) => { mousePosRef.current = { x: e.clientX, y: e.clientY }; };
    const handleMouseEnter = (e) => {
        mousePosRef.current = { x: e.clientX, y: e.clientY };
        hoverTimerRef.current = setTimeout(showTooltip, 400);
    };
    const handleMouseLeave = () => {
        clearTimeout(hoverTimerRef.current);
        setShowPreview(false);
    };

    return (
        <span
            className="relative inline-flex"
            onMouseEnter={handleMouseEnter}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
        >
            <button
                onClick={() => open(videoObj, startSeconds)}
                title={`${videoObj.name} @ ${label}`}
                className={clsx(
                    'inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-mono align-baseline',
                    'border transition-all cursor-pointer',
                    'bg-emerald-500/15 border-emerald-500/40 text-emerald-400 hover:bg-emerald-500/30 hover:border-emerald-500 active:bg-emerald-500/40 active:scale-95'
                )}
            >
                {/* Clock icon */}
                <svg className="w-3 h-3 shrink-0" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
                </svg>
                {label}
            </button>

            {showPreview && createPortal(
                <div
                    className="fixed z-[9999] w-56 rounded-xl overflow-hidden pointer-events-none"
                    style={{
                        bottom: `${window.innerHeight - tooltipPos.top}px`,
                        left: tooltipPos.left,
                        background: 'rgba(48,48,48,0.97)',
                        border: '1px solid rgba(255,255,255,0.1)',
                        boxShadow: '0 8px 32px rgba(0,0,0,0.6), 0 0 0 1px rgba(16,185,129,0.2)',
                        animation: 'chip-preview-in 0.15s ease-out both',
                    }}
                >
                    <div className="relative w-full aspect-video bg-surface-light">
                        <video
                            src={videoObj.url}
                            poster={videoObj.thumbnail ?? undefined}
                            className="w-full h-full object-cover"
                            autoPlay
                            muted
                            loop
                            playsInline
                        />
                        <div className="absolute top-0 inset-x-0 h-px bg-emerald-500/60" />
                    </div>
                    <div className="px-3 py-2">
                        <p className="text-xs font-medium text-text truncate">{videoObj.name}</p>
                        <p className="text-[10px] text-emerald-400/80 mt-0.5">⏱ {label}</p>
                    </div>
                </div>,
                document.body
            )}
        </span>
    );
}

/**
 * Wraps any string children and replaces bare ObjectIds with <VideoIdChip>.
 * Applied as a custom markdown text/p renderer so react-markdown never sees
 * modified text — no link injection needed.
 */
function renderWithChips(children) {
    if (!children) return children;
    const nodes = Array.isArray(children) ? children : [children];
    return nodes.flatMap((node, ni) => {
        if (typeof node !== 'string') return [node];
        const parts = node.split(OBJECT_ID_RE);
        return parts.map((part, pi) =>
            OBJECT_ID_RE.test(part)
                ? <VideoIdChip key={`${ni}-${pi}`} videoId={part} />
                : part
        );
    });
}

function handleCopyText(text, setCopied) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text);
    } else {
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
    }
    setCopied(true);
    toast.success("Copied to clipboard!");
    setTimeout(() => setCopied(false), 2000);
}

function InlineCode({ children, ...props }) {
    return (
        <code
            className="!bg-white/10 !px-1.5 !py-0.5 !rounded !text-[13px] !font-mono !font-normal before:!content-none after:!content-none"
            {...props}
        >
            {children}
        </code>
    );
}

function CodeBlock({ className, children }) {
    const [copied, setCopied] = useState(false);
    const match = /language-(\w+)/.exec(className || '');
    const language = match ? match[1] : 'text';

    const getCodeString = (node) => {
        if (typeof node === 'string') return node;
        if (Array.isArray(node)) return node.map(getCodeString).join('');
        if (node?.props?.children) return getCodeString(node.props.children);
        return '';
    };
    const codeString = getCodeString(children).replace(/\n$/, '');

    return (
        <div className="!relative !my-3 !rounded-lg !overflow-hidden !border !border-white/10">
            <div className="flex items-center justify-between px-4 py-2 bg-[#282c34] border-b border-white/10">
                <span className="text-xs text-gray-400 font-medium">{language}</span>
                <button
                    onClick={() => handleCopyText(codeString, setCopied)}
                    className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-white transition-colors cursor-pointer"
                    title="Copy code"
                >
                    {copied ? (
                        <>
                            <CheckIcon className="w-4 h-4" />
                            <span>Copied!</span>
                        </>
                    ) : (
                        <>
                            <ClipboardDocumentIcon className="w-4 h-4" />
                            <span>Copy code</span>
                        </>
                    )}
                </button>
            </div>
            <div className="overflow-x-auto scrollbar-thin scrollbar-thumb-surface-light scrollbar-track-transparent bg-[#282c34]">
                <SyntaxHighlighter
                    language={language}
                    style={oneDark}
                    customStyle={{
                        margin: 0,
                        padding: '1rem',
                        fontSize: '0.875rem',
                        borderRadius: 0,
                        background: 'transparent',
                        minWidth: 'fit-content',
                    }}
                    showLineNumbers={false}
                    wrapLongLines={true}
                >
                    {codeString}
                </SyntaxHighlighter>
            </div>
        </div>
    );
}

function PreBlock({ children, ...props }) {
    if (children?.type === 'code' || (children?.props?.className)) {
        return <CodeBlock {...children.props} />;
    }
    if (children?.props) {
        return <CodeBlock {...children.props} />;
    }
    return <pre {...props}>{children}</pre>;
}

export default function TextBlock({ block, role }) {
    const [copied, setCopied] = useState(false);

    return (
        <div
            className={clsx(
                'px-4 py-2 my-2 rounded-lg text-sm break-words hyphens-auto',
                role === 'user'
                    ? 'max-w-[75%] bg-accent/80 text-white self-end backdrop-blur-md shadow-lg border border-white/10 whitespace-pre-wrap'
                    : "w-full text-text self-start"
            )}
        >
            <div className="prose prose-sm prose-invert max-w-full prose-code:before:content-none prose-code:after:content-none">
                <Markdown
                    key={block.text}
                    remarkPlugins={[remarkGfm]}
                    components={{
                        code: ({ inline, className, children, ...props }) => {
                            if (!className && role === 'assistant') {
                                const text = typeof children === 'string'
                                    ? children
                                    : Array.isArray(children) ? children.join('') : '';
                                const trimmed = text.trim();

                                // Timestamp chip: `hh:mm:ss.mmm - hh:mm:ss.mmm | video_id`
                                const tsMatch = TIMESTAMP_RE.exec(trimmed);
                                if (tsMatch) {
                                    return (
                                        <TimestampChip
                                            startTs={tsMatch[1]}
                                            endTs={tsMatch[2] ?? null}
                                            videoId={tsMatch[3]}
                                        />
                                    );
                                }

                                // Plain ObjectId chip: `69d1fa…`
                                if (/^[0-9a-f]{24}$/.test(trimmed)) {
                                    return <VideoIdChip videoId={trimmed} />;
                                }
                            }
                            if (className) {
                                return <code className={className} {...props}>{children}</code>;
                            }
                            return <InlineCode {...props}>{children}</InlineCode>;
                        },
                        pre: PreBlock,
                        table: ({ children }) => (
                            <div className="overflow-x-auto scrollbar-thin scrollbar-thumb-surface-light scrollbar-track-transparent my-3">
                                <table className="min-w-full">{children}</table>
                            </div>
                        ),
                        // For assistant messages: scan ALL text-bearing elements for ObjectIds
                        ...(role === 'assistant' && {
                            p:      ({ children }) => <p>{renderWithChips(children)}</p>,
                            li:     ({ children }) => <li>{renderWithChips(children)}</li>,
                            td:     ({ children }) => <td>{renderWithChips(children)}</td>,
                            th:     ({ children }) => <th>{renderWithChips(children)}</th>,
                            strong: ({ children }) => <strong>{renderWithChips(children)}</strong>,
                            h1:     ({ children }) => <h1>{renderWithChips(children)}</h1>,
                            h2:     ({ children }) => <h2>{renderWithChips(children)}</h2>,
                            h3:     ({ children }) => <h3>{renderWithChips(children)}</h3>,
                            h4:     ({ children }) => <h4>{renderWithChips(children)}</h4>,
                            h5:     ({ children }) => <h5>{renderWithChips(children)}</h5>,
                            h6:     ({ children }) => <h6>{renderWithChips(children)}</h6>,
                        }),
                    }}
                >
                    {block.text}
                </Markdown>
            </div>
            {role === 'assistant' && (
                <button
                    onClick={() => handleCopyText(block?.text, setCopied)}
                    className="mt-3 p-1.5 rounded-md text-text-muted hover:text-text hover:bg-surface-light active:bg-surface-light transition-all duration-200 ease-in-out cursor-pointer"
                    title="Copy to clipboard"
                >
                    {copied ? (
                        <CheckIcon className="w-5 h-5 text-accent" />
                    ) : (
                        <ClipboardDocumentIcon className="w-5 h-5" />
                    )}
                </button>
            )}

        </div>
    );
}
