import { useState } from "react";
import clsx from "clsx";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { ClipboardDocumentIcon, CheckIcon } from "@heroicons/react/24/outline";
import toast from "react-hot-toast";

function handleCopyText(text, setCopied) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text);
    } else {
        // Fallback for non-secure contexts
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

// Inline code component (for `code`) - only used for inline, not block
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

// Block code component (for ```code```)
function CodeBlock({ className, children }) {
    const [copied, setCopied] = useState(false);
    const match = /language-(\w+)/.exec(className || '');
    const language = match ? match[1] : 'text';
    
    // Handle children properly - could be string, array, or nested
    const getCodeString = (node) => {
        if (typeof node === 'string') return node;
        if (Array.isArray(node)) return node.map(getCodeString).join('');
        if (node?.props?.children) return getCodeString(node.props.children);
        return '';
    };
    const codeString = getCodeString(children).replace(/\n$/, '');

    return (
        <div className="!relative !my-3 !rounded-lg !overflow-hidden !border !border-white/10">
            {/* Header */}
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
            {/* Code content with syntax highlighting */}
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

// Custom pre to pass children to CodeBlock
function PreBlock({ children, ...props }) {
    // children is the <code> element, extract its props
    if (children?.type === 'code' || (children?.props?.className)) {
        return <CodeBlock {...children.props} />;
    }
    // Fallback: if children is a code element
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
                            // If it has a className with language-, it's a code block (handled by pre)
                            // Otherwise it's inline code
                            if (className) {
                                return <code className={className} {...props}>{children}</code>;
                            }
                            return <InlineCode {...props}>{children}</InlineCode>;
                        },
                        pre: PreBlock,
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
