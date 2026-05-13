import { useState } from "react";
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

/**
 * Clickable chip for a video ObjectId. Looks up the video in workspaceVideos
 * and calls useVideoModalStore.open(video) directly — no <a> tag involved.
 */
const MINIO_BASE = 'http://100.113.186.28:9000/videos';

function VideoIdChip({ videoId }) {
    const workspaceVideos = useChatStore((s) => s.workspaceVideos);
    const open = useVideoModalStore((s) => s.open);

    // Try to find a richer record (name, thumbnail) in workspace; fall back to ID-only
    const workspaceVideo = workspaceVideos.find((v) => v._id === videoId || v.id === videoId);
    const videoObj = {
        url: `${MINIO_BASE}/${videoId}.mp4`,
        name: workspaceVideo?.name ?? videoId,
        thumbnail: workspaceVideo?.thumbnail ?? null,
    };

    return (
        <button
            onClick={() => open(videoObj)}
            title={videoObj.name}
            className={clsx(
                'inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-mono align-baseline',
                'border transition-colors cursor-pointer',
                'bg-accent/15 border-accent/40 text-accent hover:bg-accent/30 hover:border-accent'
            )}
        >
            <svg className="w-3 h-3 shrink-0" viewBox="0 0 20 20" fill="currentColor">
                <path d="M2 6a2 2 0 012-2h1v2H4a1 1 0 00-1 1v6a1 1 0 001 1h1v2H4a2 2 0 01-2-2V6zM15 4h1a2 2 0 012 2v8a2 2 0 01-2 2h-1v-2h1a1 1 0 001-1V7a1 1 0 00-1-1h-1V4zM6 4h8v2H6V4zm0 10h8v2H6v-2zm0-5h8v2H6V9z" />
            </svg>
            {videoObj.name}
        </button>
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
                            // Backtick-wrapped ObjectId → chip (e.g. `69d1fa…`)
                            if (!className && role === 'assistant') {
                                const text = typeof children === 'string'
                                    ? children
                                    : Array.isArray(children) ? children.join('') : '';
                                if (/^[0-9a-f]{24}$/.test(text.trim())) {
                                    return <VideoIdChip videoId={text.trim()} />;
                                }
                            }
                            if (className) {
                                return <code className={className} {...props}>{children}</code>;
                            }
                            return <InlineCode {...props}>{children}</InlineCode>;
                        },
                        pre: PreBlock,
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
                    className="mt-3 p-1.5 rounded-md text-text-muted hover:text-text hover:bg-surface-light transition-all duration-200 ease-in-out cursor-pointer"
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
