// type StepStatus = "done" | "active" | "pending" | "error";

// type Step = {
//     id: string;
//     title: string;
//     description?: string;
//     status: StepStatus;
//     timestamp?: string;
// };

import { useState, useEffect } from "react";
import { CheckIcon, ChevronDownIcon } from "@heroicons/react/24/solid";
import { Disclosure, DisclosureButton, DisclosurePanel } from "@headlessui/react";
import Markdown from "react-markdown";
import { ChatBubbleLeftEllipsisIcon } from "@heroicons/react/20/solid";

export default function VerticalStepper({ steps }) {
    // Track which steps are expanded - last one is always expanded
    const [expandedSteps, setExpandedSteps] = useState(new Set());

    // When steps change, collapse all except the last one
    useEffect(() => {
        setExpandedSteps(new Set([steps.length - 1]));
    }, [steps.length]);

    const toggleStep = (index) => {
        const isLast = index === steps.length - 1;
        if (isLast) return; // Last one always stays open

        setExpandedSteps((prev) => {
            const next = new Set(prev);
            if (next.has(index)) {
                next.delete(index);
            } else {
                next.add(index);
            }
            return next;
        });
    };

    return (
        <ol className="relative border-l py-2 pr-2 border-surface-light border-dashed">
            {steps.map((step, i) => {
                const isFirst = i === 0;
                const isLast = i === steps.length - 1;
                const isExpanded = isLast || expandedSteps.has(i);
                const hasDescription = !!step.description;

                return (
                    <li key={i} className="ml-6 pb-8 last:pb-0">
                        {/* Dot/Icon */}
                        {isFirst && <span
                            className={`absolute -left-3 flex h-6 w-6 items-center justify-center rounded-full border 
                            bg-accent border-accent
                            `}
                        >
                            {/* {isLast ? (
                                <svg className="h-3 w-3 text-white animate-spin" viewBox="0 0 24 24" fill="none">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                                </svg>
                            ) : ( */}
                                {/* <CheckIcon className="h-3 w-3 text-white" /> */}
                                {/* thinking bubble icon */}
                                <ChatBubbleLeftEllipsisIcon className="h-3 w-3 text-white"/>
                            {/* )} */}
                        </span>}

                        {/* Content */}
                        <div>
                            <button
                                type="button"
                                onClick={() => toggleStep(i)}
                                className={`flex items-center gap-1 text-left ${hasDescription && !isLast ? "cursor-pointer" : "cursor-default"}`}
                                disabled={isLast || !hasDescription}
                            >
                                <h3 className="text-sm font-medium text-text">{step.title}</h3>
                                {hasDescription && !isLast && (
                                    <ChevronDownIcon
                                        className={`h-4 w-4 text-text-muted transition-transform duration-200 ${isExpanded ? "rotate-180" : ""}`}
                                    />
                                )}
                            </button>
                            {hasDescription && (
                                <div
                                    className={`overflow-hidden transition-all duration-200 ${isExpanded ? " opacity-100" : "max-h-0 opacity-0"}`}
                                >
                                    <div className="mt-1 text-sm text-text-muted">
                                        <Markdown>
                                            {step.description}
                                        </Markdown>
                                    </div>
                                </div>
                            )}
                        </div>
                    </li>
                );
            })}
        </ol>
    );
}
