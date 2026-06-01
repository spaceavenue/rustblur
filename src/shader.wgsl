struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) texel_coords: vec2<f32>
};

struct BlurParams {
    halfpixel: vec2<f32>,
    offset: f32,
    _padding: u32
};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {

    // let x = f32(1 - i32(in_vertex_index)) * 3.0;
    // let y = f32(i32(in_vertex_index & 1u) * 4 - 1);

    //easier to understand
    let pos = array(
        vec2f(-1.0, 3.0),
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0)
    );
    var out: VertexOutput;

    out.clip_position = vec4f(pos[in_vertex_index], 0.0, 1.0);
    out.texel_coords = vec2f(pos[in_vertex_index].x * 0.5 + 0.5, 1.0 - (pos[in_vertex_index].y * 0.5 + 0.5));

    return out;
}

@fragment
@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;
@group(0) @binding(2)
var<uniform> params: BlurParams;

@fragment
fn fs_down(in: VertexOutput) -> @location(0) vec4<f32> {
    /* Dual Kawase downsample: sample center + 4 diagonal corners */

    let uv = vec2f(in.texel_coords);
    let o = vec2f(params.halfpixel * params.offset);
	
	// center with 4x weight
	var color = textureSample(t_diffuse, s_diffuse, uv) * 4;
	
	// 4 diagonal corners with 1x weight
	color += textureSample(t_diffuse, s_diffuse, uv + vec2(-o.x, -o.y)); /* bottom-left */
	color += textureSample(t_diffuse, s_diffuse, uv + vec2( o.x, -o.y)); /* bottom-right   */
	color += textureSample(t_diffuse, s_diffuse, uv + vec2(-o.x,  o.y)); /* top-left */
	color += textureSample(t_diffuse, s_diffuse, uv + vec2( o.x,  o.y)); /* top-right */
	
	// normalize by total weight (8)
    return (color / 8.0);
}

@fragment
fn fs_up(in: VertexOutput) -> @location(0) vec4<f32> {
    /* Dual Kawase upsample: sample corners + 4 pixels */

    let uv = vec2f(in.texel_coords);
    let o = vec2f(params.halfpixel * params.offset);
    var color = vec4f(0.0);

    // 4 edges with 1x weight
	color += textureSample(t_diffuse, s_diffuse, uv + vec2( o.x * 2.0, 0 ));
	color += textureSample(t_diffuse, s_diffuse, uv + vec2( 0, o.y * 2.0 ));
	color += textureSample(t_diffuse, s_diffuse, uv + vec2( -o.x * 2.0, 0 ));
	color += textureSample(t_diffuse, s_diffuse, uv + vec2( 0, -o.y * 2.0 ));

    // 4 diagonal corners with 2x weight
	color += textureSample(t_diffuse, s_diffuse, uv + vec2(-o.x, -o.y)) * 2.0;
	color += textureSample(t_diffuse, s_diffuse, uv + vec2( o.x, -o.y)) * 2.0;
	color += textureSample(t_diffuse, s_diffuse, uv + vec2(-o.x,  o.y)) * 2.0;
	color += textureSample(t_diffuse, s_diffuse, uv + vec2( o.x,  o.y)) * 2.0;

	// normalize by total weight (12)
    return (color / 12.0);
}