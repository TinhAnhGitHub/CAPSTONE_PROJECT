import ErrorBoundary from "@/components/common/errors/ErrorBoundary";
import PrivateHome from "@/pages/PrivateHome";
const privateRoutes = [
    {
        path: "private",
        element: <PrivateHome />,
        errorElement: <ErrorBoundary />,
    },
];

export default privateRoutes;
