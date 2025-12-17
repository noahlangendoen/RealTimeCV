#include "core/env_loader.hpp"
#include "pipeline/pipeline_manager.hpp"

// Modify the path to the env file as needed
#define ENV_PATH "../.env"

int main() {

    EnvLoader env;

    if (!env.load(ENV_PATH)) {
        std::cerr << "ERROR: .env file not found";
        return -1;
    }

    std::string detector_path = env.get("FACE_DETECTOR");
    std::string classifier_path = env.get("BEST_CLASSIFIER");

    PipelineManager pipeline(
        detector_path,
        classifier_path,
        0,
        640,
        480
    );
    
    if (!pipeline.initialize()) {
        std::cerr << "Failed to initialize pipeline" << std::endl;
        return -1;
    }

    pipeline.start();

    while (pipeline.isRunning()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "Pipeline stopped" << std::endl;
    
    return 0;
}