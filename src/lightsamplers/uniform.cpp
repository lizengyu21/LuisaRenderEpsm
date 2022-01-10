//
// Created by Mike Smith on 2022/1/10.
//

#include <luisa-compute.h>
#include <scene/light_sampler.h>
#include <scene/pipeline.h>

namespace luisa::render {

class UniformLightSampler final : public LightSampler {

public:
    unique_ptr<Instance> build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept override;
    UniformLightSampler(Scene *scene, const SceneNodeDesc *desc) noexcept : LightSampler{scene, desc} {}
    [[nodiscard]] string_view impl_type() const noexcept override { return "uniform"; }
};

class UniformLightSamplerInstance final : public LightSampler::Instance {

private:
    const Pipeline &_pipeline;
    uint _light_to_instance_buffer_id{};

public:
    UniformLightSamplerInstance(const LightSampler *sampler, Pipeline &pipeline, CommandBuffer &command_buffer) noexcept
        : LightSampler::Instance{sampler},
          _pipeline{pipeline} {
        auto [view, buffer_id] = pipeline.arena_buffer<uint>(pipeline.lights().size());
        _light_to_instance_buffer_id = buffer_id;
        luisa::vector<uint> light_to_instance_id(pipeline.lights().size());
        std::transform(
            pipeline.lights().cbegin(), pipeline.lights().cend(),
            light_to_instance_id.begin(),
            [](auto light) noexcept { return std::get<0>(light.second); });
        command_buffer << view.copy_from(light_to_instance_id.data())
                       << compute::commit();// lifetime
    }
    void update(Stream &) noexcept override {}
    [[nodiscard]] Float pdf(const Interaction &it) const noexcept override {
        return static_cast<float>(1.0 / static_cast<double>(_pipeline.lights().size()));
    }
    [[nodiscard]] LightSampler::Selection sample(Sampler::Instance &sampler, const Interaction &it) const noexcept override {
        using namespace luisa::compute;
        auto u = sampler.generate_1d();
        auto n = static_cast<uint>(_pipeline.lights().size());
        auto i = clamp(cast<uint>(u * static_cast<float>(n)), 0u, n - 1u);
        auto instance_id = _pipeline.buffer<uint>(_light_to_instance_buffer_id).read(i);
        return {.inst = instance_id, .pdf = pdf(it)};
    }
};

unique_ptr<LightSampler::Instance> UniformLightSampler::build(Pipeline &pipeline, CommandBuffer &command_buffer) const noexcept {
    return luisa::make_unique<UniformLightSamplerInstance>(this, pipeline, command_buffer);
}

}// namespace luisa::render

LUISA_RENDER_MAKE_SCENE_NODE_PLUGIN(luisa::render::UniformLightSampler)