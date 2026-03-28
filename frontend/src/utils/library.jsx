export const errorIngested = (ingested_status) => {
    return  ingested_status === -1;
}
export const ingested = (ingested_status) => {
    return ingested_status === 100;
}
