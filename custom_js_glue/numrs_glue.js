export function asarray(data) {
    if (typeof(data[0]) === 'number') {
        return asarray1d(data);
    } else if (typeof(data[0][0]) === 'number') {
        return asarray2d(data);
    } else if (typeof(data[0][0][0]) === 'number') {
        return asarray3d(data);
    } else if (typeof(data[0][0][0][0]) === 'number') {
        return asarray4d(data);
    }
}