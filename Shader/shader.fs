#version 430
varying mediump vec2 texCoords;

uniform vec4 colors;
uniform sampler2D texture;

void main(void)
{
    vec4 tex = texture(texture, texCoords.xy);

    if(tex.rgb == vec3(255.f / 255.f, 155.0f / 255.f, 33.0f / 255.0f))
        tex = colors;

    gl_FragColor = tex;
}