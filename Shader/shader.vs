attribute mediump vec3 aPos;
attribute mediump vec2 aTexCoord;
attribute mediump vec4 aColors;

varying mediump vec2 texCoords;
varying mediump vec4 colors;
uniform mat4 mvp;

void main(void)
{
    gl_Position = mvp * vec4(aPos, 1.0); 
    texCoords = vec2(aTexCoord.x, aTexCoord.y);
    colors = aColors;
}