#include <algorithm>  // for std::clamp
#include <cstdint>    // for uint8_t
#include <array>      // for std::array
#include <cmath>      // maybe for rounding (if needed)

//-----------------------------------------------------
// Pixel struct & image class as given by the exercise
//-----------------------------------------------------
struct pixel {
public:
    uint8_t red, green, blue;
    uint8_t alpha;  // 255 => opaque, 0 => transparent
};

// Suppose we have a templated image class like this:
// template <typename T, size_t N, size_t M>
// class image { ... };
// For brevity, we use the 'alpha_image' as specified:
static const size_t height = 128;
static const size_t width  = 128;
template <typename T, size_t N, size_t M>
class image {
    using storage_type = std::array<std::array<T, M>, N>;
public:
    image() = default;

    T& operator()(size_t i, size_t j) {
        return _array[i][j];
    }
    T operator()(size_t i, size_t j) const {
        return _array[i][j];
    }
private:
    storage_type _array;
};

using alpha_image = image<pixel, height, width>;

//-----------------------------------------
// Alpha-over operator (sequential version)
//-----------------------------------------
alpha_image alpha_over_operator(const alpha_image& f, const alpha_image& b)
{
    alpha_image out;

    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            pixel fp = f(i, j);  // foreground pixel
            pixel bp = b(i, j);  // background pixel
            pixel op;            // output pixel

            // Convert alpha to [0..1] range
            float af = static_cast<float>(fp.alpha) / 255.0f;
            float ab = static_cast<float>(bp.alpha) / 255.0f;

            // Compute output alpha
            float ao = af + ab * (1.0f - af);

            // If ao == 0, the result is fully transparent
            if (ao > 0.0f) {
                // For each color channel (r,g,b):
                float rf = static_cast<float>(fp.red);
                float gf = static_cast<float>(fp.green);
                float bf = static_cast<float>(fp.blue);

                float rb = static_cast<float>(bp.red);
                float gb = static_cast<float>(bp.green);
                float bb = static_cast<float>(bp.blue);

                float ro = (rf * af + rb * ab * (1.0f - af)) / ao;
                float go = (gf * af + gb * ab * (1.0f - af)) / ao;
                float bo = (bf * af + bb * ab * (1.0f - af)) / ao;

                // Clamp and convert back to uint8_t
                op.red   = static_cast<uint8_t>(std::clamp(ro, 0.0f, 255.0f));
                op.green = static_cast<uint8_t>(std::clamp(go, 0.0f, 255.0f));
                op.blue  = static_cast<uint8_t>(std::clamp(bo, 0.0f, 255.0f));
            }
            else {
                // Completely transparent => color = 0
                op.red   = 0;
                op.green = 0;
                op.blue  = 0;
            }

            // Convert alpha in [0..1] back to [0..255]
            float alphaOut = ao * 255.0f;
            op.alpha = static_cast<uint8_t>(std::clamp(alphaOut, 0.0f, 255.0f));

            out(i, j) = op;
        }
    }
    return out;
}


// Parallel version of the alpha-over operator data based (becauso of regular problem)

#include <thread>
#include <vector>
#include <mutex>


alpha_image alpha_over_operator_parallel(const alpha_image& f, const alpha_image& b, size_t num_threads) {
    alpha_image out;
    std::mutex mtx;

    // process_chunk is a lambda function that processes a chunk of rows
    auto process_chunk = [&](size_t start_row, size_t end_row) {
        for (size_t i = start_row; i < end_row; ++i) {
            for (size_t j = 0; j < width; ++j) {
                pixel fp = f(i, j);  // foreground pixel
                pixel bp = b(i, j);  // background pixel
                pixel op;            // output pixel

                // Convert alpha to [0..1] range
                float af = static_cast<float>(fp.alpha) / 255.0f;
                float ab = static_cast<float>(bp.alpha) / 255.0f;

                // Compute output alpha
                float ao = af + ab * (1.0f - af);

                // If ao == 0, the result is fully transparent
                if (ao > 0.0f) {
                    // For each color channel (r,g,b):
                    float rf = static_cast<float>(fp.red);
                    float gf = static_cast<float>(fp.green);
                    float bf = static_cast<float>(fp.blue);

                    float rb = static_cast<float>(bp.red);
                    float gb = static_cast<float>(bp.green);
                    float bb = static_cast<float>(bp.blue);

                    float ro = (rf * af + rb * ab * (1.0f - af)) / ao;
                    float go = (gf * af + gb * ab * (1.0f - af)) / ao;
                    float bo = (bf * af + bb * ab * (1.0f - af)) / ao;

                    // Clamp and convert back to uint8_t
                    op.red   = static_cast<uint8_t>(std::clamp(ro, 0.0f, 255.0f));
                    op.green = static_cast<uint8_t>(std::clamp(go, 0.0f, 255.0f));
                    op.blue  = static_cast<uint8_t>(std::clamp(bo, 0.0f, 255.0f));
                } else {
                    // Completely transparent => color = 0
                    op.red   = 0;
                    op.green = 0;
                    op.blue  = 0;
                }

                // Convert alpha in [0..1] back to [0..255]
                float alphaOut = ao * 255.0f;
                op.alpha = static_cast<uint8_t>(std::clamp(alphaOut, 0.0f, 255.0f));

                // Update the output image
                std::lock_guard<std::mutex> lock(mtx);
                out(i, j) = op;
            }
        }
    };

    std::vector<std::thread> threads;
    size_t chunk_size = height / num_threads;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start_row = t * chunk_size;
        size_t end_row = (t == num_threads - 1) ? height : start_row + chunk_size;
        threads.emplace_back(process_chunk, start_row, end_row);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return out;
}


// With region structure and thread pool

struct Region {
    int x0, x1, y0, y1;
    explicit Region(int x0, int x1, int y0, int y1) :
        x0(x0), x1(x1), y0(y0), y1(y1) {}
    void print() {
      std::cout << "x0: " << x0 << ", x1: " << x1 << std::endl;
      std::cout << "y0: " << y0 << ", y1: " << y1 << std::endl;
      std::cout << std::endl;
    }
};

template <typename T>
void process_region(const image<T, 128, 128>& f, const image<T, 128, 128>& b, image<T, 128, 128>& out, const Region& reg, std::mutex& mtx) {
    for (int i = reg.y0; i < reg.y1; ++i) {
        for (int j = reg.x0; j < reg.x1; ++j) {
            pixel fp = f(i, j);  // foreground pixel
            pixel bp = b(i, j);  // background pixel
            pixel op;            // output pixel

            // Convert alpha to [0..1] range
            float af = static_cast<float>(fp.alpha) / 255.0f;
            float ab = static_cast<float>(bp.alpha) / 255.0f;

            // Compute output alpha
            float ao = af + ab * (1.0f - af);

            // If ao == 0, the result is fully transparent
            if (ao > 0.0f) {
                // For each color channel (r,g,b):
                float rf = static_cast<float>(fp.red);
                float gf = static_cast<float>(fp.green);
                float bf = static_cast<float>(fp.blue);

                float rb = static_cast<float>(bp.red);
                float gb = static_cast<float>(bp.green);
                float bb = static_cast<float>(bp.blue);

                float ro = (rf * af + rb * ab * (1.0f - af)) / ao;
                float go = (gf * af + gb * ab * (1.0f - af)) / ao;
                float bo = (bf * af + bb * ab * (1.0f - af)) / ao;

                // Clamp and convert back to uint8_t
                op.red   = static_cast<uint8_t>(std::clamp(ro, 0.0f, 255.0f));
                op.green = static_cast<uint8_t>(std::clamp(go, 0.0f, 255.0f));
                op.blue  = static_cast<uint8_t>(std::clamp(bo, 0.0f, 255.0f));
            } else {
                // Completely transparent => color = 0
                op.red   = 0;
                op.green = 0;
                op.blue  = 0;
            }

            // Convert alpha in [0..1] back to [0..255]
            float alphaOut = ao * 255.0f;
            op.alpha = static_cast<uint8_t>(std::clamp(alphaOut, 0.0f, 255.0f));

            // Update the output image
            std::lock_guard<std::mutex> lock(mtx);
            out(i, j) = op;
        }
    }
}

template <typename T>
image<T, 128, 128> alpha_over_operator_parallel(const image<T, 128, 128>& f, const image<T, 128, 128>& b, size_t num_threads) {
    if (f.height != b.height || f.width != b.width) {
        throw std::runtime_error("Images must have the same dimensions.");
    }

    image<T, 128, 128> out;
    std::mutex mtx;

    size_t w_div = 4;  // Number of divisions along the width
    size_t h_div = 4;  // Number of divisions along the height

    size_t region_width = f.width / w_div;
    size_t region_height = f.height / h_div;

    thread_pool pool(num_threads);

    for (size_t i = 0; i < w_div; ++i) {
        for (size_t j = 0; j < h_div; ++j) {
            // Define the region
            int x0 = i * region_width;
            int x1 = (i + 1) * region_width;
            int y0 = j * region_height;
            int y1 = (j + 1) * region_height;

            Region reg(x0, x1, y0, y1);

            // Submit the task to the thread pool
            pool.submit([&, reg]() {
                process_region(f, b, out, reg, mtx);
            });
        }
    }

    // Wait for completion
    pool.wait();

    return out;
}