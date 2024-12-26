#include "CImg.h"
#include <vector>
#include <iostream>
#include <string>

using namespace cimg_library;

struct ImageFragment {
    unsigned char* data; // Pointer to fragment data
    int width;           // Width of the fragment
    int height;          // Height of the fragment
    int x_offset;        // X-offset in the original image
    int y_offset;        // Y-offset in the original image
};


// Function to LOAD a base image and simulate a set of images
CImg<unsigned char> simulate_large_image(const std::string& image_path, int num_replicas) {
    
    // Load the base image
    CImg<unsigned char> base_image(image_path.c_str());
    int width = base_image.width();
    int height = base_image.height();

    // Simulate a large image by replicating the base image
    CImg<unsigned char> simulated_image(width, height * num_replicas, 1, 1, 0);
    for (int i = 0; i < num_replicas; i++) {
        simulated_image.draw_image(0, i * height, 0, 0, base_image);
    }

    std::cout << "Simulated a large image with " << num_replicas << " replicas of the base image.\n";
    return simulated_image;
}


// Function to EXTRACT a specific replica from the simulated image
CImg<unsigned char> extract_image(const CImg<unsigned char>& simulated_image, int replica_index, int width, int height) {
    
    // Index the region corresponding to the desired replica
    int y_start = replica_index * height;
    int y_end = (replica_index + 1) * height - 1;

    // Extract the region corresponding to the desired replica
    CImg<unsigned char> extracted_image = simulated_image.get_crop(0, y_start, width - 1, y_end);

    std::cout << "Extracted replica " << replica_index << " from vertical stack.\n";
    return extracted_image;
}

// Function to DIVIDE an image into fragments
std::vector<ImageFragment> divide_image_into_fragments(const CImg<unsigned char>& image, int num_fragments) {
    
    // Calculate the height of each fragment
    // We choose height since we have stacked the replicas vertically
    // Last fragment may have a different height (We must take this into account on the kernel)
    int width = image.width();
    int height = image.height();
    int fragment_height = height / num_fragments;

    std::vector<ImageFragment> fragments;
    for (int i = 0; i < num_fragments; i++) {
        int y_start = i * fragment_height;
        int y_end = (i == num_fragments - 1) ? height : (i + 1) * fragment_height;

        ImageFragment fragment;
        fragment.data = image.data(0, y_start); // Pointer to the fragment's starting pixel
        fragment.width = width;
        fragment.height = y_end - y_start;
        fragment.x_offset = 0;
        fragment.y_offset = y_start;

        fragments.push_back(fragment);
        std::cout << "Fragment " << i << ": Height = " << fragment.height
                  << ", Y-offset = " << fragment.y_offset << "\n";
    }

    return fragments;
}


int main() {
    // Path to the base image
    std::string base_image_path = "../images/input/base_image.jpg";

    // Simulate a large image with replicas
    int num_replicas = 5000;
    CImg<unsigned char> simulated_image = simulate_large_image(base_image_path, num_replicas);

    // Divide the large image into fragments
    int num_fragments = 2; // One fragment per GPU
    auto fragments = divide_image_into_fragments(simulated_image, num_fragments);

    // Extract a specific replica for testing
    int replica_index = 9; // Example: Extract the 10th replica
    CImg<unsigned char> replica_image = extract_image(simulated_image, replica_index, simulated_image.width(), simulated_image.height() / num_replicas);
    replica_image.save("replica_10.jpg");

    // Placeholder: Further processing for each fragment
    for (int i = 0; i < fragments.size(); i++) {
        std::cout << "Processing fragment " << i << "...\n";
        // Add processing logic here (e.g., send to GPU)
    }

    return 0;
}
