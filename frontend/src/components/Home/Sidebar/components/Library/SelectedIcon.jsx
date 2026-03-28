import { CheckCircleIcon } from "@heroicons/react/24/solid"
import { PlusCircleIcon } from "@heroicons/react/24/outline"

export default function SelectedIcon({ selected }) {
    return (
        selected
            ? <CheckCircleIcon className="size-5 text-accent" />
            : <PlusCircleIcon className="size-5 text-white/70 hover:text-white transition-colors" />
    )
}
