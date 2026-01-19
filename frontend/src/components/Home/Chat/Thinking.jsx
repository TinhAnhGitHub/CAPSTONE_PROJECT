import VerticalStepper from '@/components/common/components/VerticalStepper'
import { Disclosure, DisclosureButton, DisclosurePanel } from '@headlessui/react'
import { ChevronDownIcon } from '@heroicons/react/20/solid'

export default function Thinking({ thinking = [{
    title: "Drafting the Response",
    description: "I'm now putting together the response, considering all the parameters. I am structuring it to meet the requirements precisely. My focus is on concise and clear communication."
},
{
    title: "Summarizing This Response",
    description: "I'm now focused on distilling the key element of the current response. It seems relatively straightforward, but I'm being precise. The goal is a concise summary of the content, following specific instructions."
}] }) {

    return (
        <div className="w-full px-4">
            <div className="w-full max-w-lg divide-y divide-surface-light rounded-r-xl">
                <VerticalStepper steps={thinking} />
            </div>
        </div>
    )
}
