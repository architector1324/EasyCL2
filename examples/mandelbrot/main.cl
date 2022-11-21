kernel void mandelbrot(global uchar* data, uint w, uint h, float px, float py, float mag, uint maxIter){
    float aspect = w / h;

    uint x = get_global_id(0);
    uint y = get_global_id(1);

    float i = ((float)x - w / 2) / (mag * w / 4) - px;
    float j = ((float)y - h / 2) / (mag * h * aspect / 4) - py;

    float oldI = i;
    float oldJ = j;

    uint k = 0;

    for(; k < maxIter; k++) {
        float a = i * i - j * j;
        float b = 2 * i * j;
        i = a + oldI;
        j = b + oldJ;

        if(i * i + j * j > 4) break;
    }

    uint value = 255 * k / maxIter;

    data[3 * (x + w * y)] = value;
    data[3 * (x + w * y) + 1] = value;
    data[3 * (x + w * y) + 2] = value;
}
