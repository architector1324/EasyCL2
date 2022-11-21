kernel void mandelbrot(global uchar* data, uint w, uint h, uint maxIter){
    float aspect = w / h;

    size_t x = get_global_id(0);
    size_t y = get_global_id(1);

    float i = ((float)x - w / 2) / (w / 4) - 0.65f;
    float j = ((float)y - h / 2) / (h * aspect / 4);

    float oldI = i;
    float oldJ = j;

    size_t k = 0;

    for(; k < maxIter; k++) {
        float a = i * i - j * j;
        float b = 2 * i * j;
        i = a + oldI;
        j = b + oldJ;

        if(i * i + j * j > 4) break;
    }

    size_t value = 255 * k / maxIter;

    data[3 * (x + w * y)] = value;
    data[3 * (x + w * y) + 1] = value;
    data[3 * (x + w * y) + 2] = value;
}
