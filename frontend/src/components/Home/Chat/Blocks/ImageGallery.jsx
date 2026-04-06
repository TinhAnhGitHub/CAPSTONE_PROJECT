import { useState } from "react";
import clsx from "clsx";
import { ChevronDownIcon, PhotoIcon } from "@heroicons/react/24/solid";
import ImageViewer from "../ImageViewer";

export default function ImageGallery({ allImages, imageCount, previewCount = 6 }) {
    const [isExpanded, setIsExpanded] = useState(false);

    const hasMany = imageCount > previewCount;
    const displayImages = isExpanded ? allImages : allImages.slice(0, previewCount);
    const hiddenCount = imageCount - previewCount;

    return (
        <div className="max-w-full self-start px-4 py-2">
            {/* Header with count and toggle */}
            {hasMany && (
                <button
                    onClick={() => setIsExpanded(!isExpanded)}
                    className="flex items-center gap-2 mb-2 px-2 py-1 text-sm text-text-muted hover:text-text transition-colors"
                >
                    <PhotoIcon className="w-4 h-4" />
                    <span>{imageCount} images</span>
                    <span className="text-text-dim">•</span>
                    <span className="text-accent">{isExpanded ? 'Collapse' : 'Show all'}</span>
                    <ChevronDownIcon
                        className={clsx(
                            "w-4 h-4 transition-transform duration-200",
                            isExpanded && "rotate-180"
                        )}
                    />
                </button>
            )}

            {/* Flex wrap grid */}
            <div className="overflow-x-auto scrollbar-thin scrollbar-thumb-surface-light scrollbar-track-transparent pb-2">
                <div className="flex flex-wrap gap-2 max-w-[660px]">
                    {displayImages.map((img, i) => (
                        <div key={`${i}-${img.url}`} className="w-[100px] aspect-video">
                            <ImageViewer
                                image={img}
                                images={allImages}
                                startIndex={i}
                                className="w-full h-full object-cover rounded-lg"
                            />
                        </div>
                    ))}

                    {/* Show "+X more" tile if collapsed and has more */}
                    {!isExpanded && hasMany && (
                        <button
                            onClick={(e) => {
                                e.stopPropagation();
                                setIsExpanded(true);
                            }}
                            className="w-[100px] aspect-video flex items-center justify-center bg-surface hover:bg-surface-hover rounded-lg text-text-muted hover:text-text transition-colors cursor-pointer"
                        >
                            <span className="text-lg font-medium">+{hiddenCount}</span>
                        </button>
                    )}
                </div>
            </div>
        </div>
    );
}
