import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { image } = body;

    // Here you would typically send the image data to your Python backend
    // For now, let's just return a mock response
    return NextResponse.json({ arabic_char: 'أ', english_name: 'Alif' });
  } catch (error) {
    console.error('Classification error:', error);
    return NextResponse.json({ error: 'Classification failed' }, { status: 500 });
  }
}