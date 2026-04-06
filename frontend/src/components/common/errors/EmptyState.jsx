// components/common/EmptyState.jsx

import { useTranslation } from "react-i18next";

export default function EmptyState({ message }) {
    const { t } = useTranslation("common");
    message = message || t("empty_state");
    return (
        <div className="text-center py-10 text-white">
            <p className="text-xl">{message}</p>
        </div>
    );
}
