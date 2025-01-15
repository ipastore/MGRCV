struct pixel {
    public:
        uint8_t r, g, b;
        uint8_t a;
};

template <typename T, size_t N, size_t M>
class image {
    using storage_type = std::array<std::array<T, M>, N>;
    private:
        storage_type _array;
    public:
        image(){};
        T& operator()(size_t i, size_t j) {
            return _array[i][j];
        };
        T operator()(size_t i, size_t j) const {
            return _array[i][j];
        };
};

const size_t height = 128, width = 128;
using alpha_image = image<pixel, height, width>;

// Ejercicio 1.a
alpha_image alpha_image_over_operator(const alpha_image& f, const alpha_image& b) {

    alpha_image result;

    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            pixel pf = f(i, j);
            pixel pb = b(i, j);
            result(i, j) = realizar_trabajo(pf, pb);
        }
    }
}

pixel realizar_trabajo(pixel pf, pixel pb){

    float af = static_cast<float>(pf.a) / 255.0f;
    float ab = static_cast<float>(pb.a) / 255.0f;

    float ao = af + ab * ( 1- af);

    pixel po;


    if(ao > 0.0f){
        float rf = static_cast<float>(pf.r);
        float gf = static_cast<float>(pf.g);
        float bf = static_cast<float>(pf.b);

        float rb = static_cast<float>(pb.r);
        float gb = static_cast<float>(pb.g);
        float bb = static_cast<float>(pb.b);

        float ro = (rf * af + rb * ab * (1 - af)) / ao ;
        float go = (gf * af + gb * ab * (1 - af)) / ao ;
        float bo = (bf * af + bb * ab * (1 - af)) / ao ;

        po.r = static_cast<uint8_t>(std::clamp(ro, 0.0f, 255.0f));
        po.g = static_cast<uint8_t>(std::clamp(go, 0.0f, 255.0f));
        po.b = static_cast<uint8_t>(std::clamp(bo, 0.0f, 255.0f));
        po.a =  static_cast<uint8_t>(std::clamp(ao * 255.0f , 0.0f, 255.0f));

    }else{
        po.r = 0;
        po.g = 0;
        po.b = 0;
        po.a = ao;

    }
    return po;
} 

// Ejercicio 1.a
alpha_image alpha_image_over_operator_parallel(alpha_image& result, const alpha_image& f, const alpha_image& b, size_t start_pixel, size_t end_pixel){

    for( ; start_pixel < end_pixel; start_pixel++){
        int i = start_pixel / width;
        int j = start_pixel % width;
        pixel pf = f(i, j);
        pixel pb = b(i, j);
        result(i, j) = realizar_trabajo(pf, pb);
    }


}

alpha_image alpha_image_over_operator(const alpha_image& f, const alpha_image& b) {

    alpha_image result;

    size_t num_threads = std::thread::hardware_concurrency();

    const size_t chunk_size = width*height / num_threads;
    const size_t remaining_pixel = width*height % num_threads;

    std::vector<std::thread> thread_pool;

    size_t start_pixel = 0;

    for(int i = 0; i < num_threads; i++){

        size_t end_pixel = start_pixel + chunk_size + (i < remaining_pixel) ? 1 : 0;

        thread_pool.emplace_back(alpha_image_over_operator_parallel, std::ref(result),f,b, start_pixel, end_pixel);

        start_pixel = end_pixel;
    }

        // Wait for all threads to complete
    for (auto &t : thread_pool) {
        t.join();
    }

    return result;

}