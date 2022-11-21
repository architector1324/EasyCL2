kernel void test(global int* a, int b){
    size_t i = get_global_id(0);
    a[i] = b * ((int)get_group_id(0) + 1);
}
