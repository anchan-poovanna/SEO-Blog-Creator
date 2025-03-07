import streamlit as st
from analyzer import LLMEnhancedAnalyzer
from serp import get_search_results
from perplexity import deep_research
import re
import anthropic
import os
from dotenv import load_dotenv
from fpdf import FPDF
import io
from datetime import datetime

# Load environment variables
load_dotenv()

# Get API keys from environment variables
FIRECRAWL_API_KEY = st.secrets.get("FIRECRAWL_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")

def extract_research_content(research_text):
    """Extract the content from the research report, removing the thinking process."""
    try:
        # Remove the thinking process part
        if '<think>' in research_text and '</think>' in research_text:
            think_pattern = r'<think>.*?</think>'
            research_text = re.sub(think_pattern, '', research_text, flags=re.DOTALL)
        
        # Clean up any remaining tags or indicators
        research_text = research_text.replace('=== Research Report ===', '')
        
        # Remove the competitor content analysis if present
        if '--- Competitor Content Analysis Summary ---' in research_text:
            split_text = research_text.split('--- Competitor Content Analysis Summary ---')
            research_text = split_text[0].strip()
        
        return research_text.strip()
    except Exception as e:
        print(f"Error extracting research content: {e}")
        return research_text

def generate_blog_content(outline_content, research_content=None):
    """Generate blog content using Claude API"""
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    
    client = anthropic.Anthropic(
        api_key=ANTHROPIC_API_KEY
    )
    
    prompt = f"""
Create a blog based on the outline document attached. Follow every detail, including the word count, content type, and structural guidelines provided. The target audience (ICP) are- people in the construction industry, people looking for information in the construction industry, construction company owners and related personas who want to know more about the industry and the policies.

AI SEO Guidelines to Follow:
Use Conversational, Natural Language
Write content that mirrors how users ask questions.
Incorporate long-tail and question-based keywords naturally.
Optimize for semantic search by focusing on topics, not just keywords.
Pro Tip: Use tools like AnswerThePublic to find search queries in this niche.

Optimize for AI Overviews
Structure content using Q&A formats with direct, concise answers.
Format key sections with bullet points and numbered lists for readability.
Implement FAQ schema markup to improve AI searchability.
Pro Tip: Research Google's People Also Ask section for top-performing questions.

EEAT (Expertise, Experience, Authority, Trustworthiness)
Include a detailed author bio showcasing industry credentials.
Link to credible sources such as studies, case studies, and expert opinions.
Build backlinks from trusted domains to enhance authority.
Pro Tip: Use first-hand data and case studies to stand out from AI-generated content.

Improve Structured Data & Schema
Implement FAQ, How-To, and Article schema for better AI visibility.
Add author markup to highlight expert-written content.
Use breadcrumbs and internal linking to strengthen content relationships.
Pro Tip: Validate structured data using Google's Rich Results Test.

Answer in the First Paragraph
AI-driven search often pulls direct answers from well-structured content.
Include key information in the first paragraph for better indexing.
Pro Tip: Check the research insights to refine the clarity of content.

Expand Content with AI Search Queries
Analyze AI-generated search responses for subtopic ideas.
Create content that fills information gaps in existing resources.
Expand blog sections based on AI-recommended queries.
Ensure the word count falls within the specified range.
Maintain a professional tone while keeping the content engaging.
Incorporate relevant statistics, data, and expert quotes where applicable.
Add visual elements (charts, infographics, etc.) to improve readability.
Include internal links to related AI content (advancements, funding strategies, startup guidance).
External links should point to authoritative sources (AI research, industry reports, expert case studies).
The content format should be as described in the outline in the article type part as it aligns with user preferences and search behavior.
Deliver a fully optimized, structured, and AI-friendly blog that aligns with all outlined requirements.

Here is the outline document:

{outline_content}
"""
    
    if research_content:
        prompt += f"""
Additionally, here is supplementary research that contains valuable insights and information about the topic:

{research_content}
"""
    
    prompt += """
Based on the provided outline and research, generate a complete blog article that follows all the guidelines provided.
"""
    
    try:
        # Create a message with Claude
        # Note: Removed the 'thinking' parameter as it's not supported in the current SDK version
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=4096,
            temperature=0.7,
            system="You are an expert blog writer and SEO specialist. Your task is to generate high-quality, SEO-optimized blog content based on the given outline.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        if not message or not message.content:
            raise ValueError("No content received from Claude API")
            
        return message.content[0].text
    
    except Exception as e:
        raise Exception(f"Error generating content: {str(e)}")

def format_blog_content(content):
    """Format blog content with enhanced header styling and spacing"""
    lines = content.split('\n')
    formatted_lines = []
    
    for line in lines:
        # Handle H1 headers (usually the title)
        if line.startswith('# '):
            formatted_lines.append(f"<h1 style='font-size: 2.5em; color: #ffffff; margin-top: 1.5em; margin-bottom: 0.8em; font-weight: 700;'>{line[2:]}</h1>")
        # Handle H2 headers (main sections)
        elif line.startswith('## '):
            formatted_lines.append(f"<h2 style='font-size: 2em; color: #ffffff; margin-top: 1.5em; margin-bottom: 0.6em; font-weight: 600;'>{line[3:]}</h2>")
        # Handle H3 headers (subsections)
        elif line.startswith('### '):
            formatted_lines.append(f"<h3 style='font-size: 1.5em; color: #ffffff; margin-top: 1.2em; margin-bottom: 0.4em; font-weight: 500;'>{line[4:]}</h3>")
        # Handle paragraphs
        else:
            formatted_lines.append(f"<p style='font-size: 1.1em; line-height: 1.6; margin-bottom: 1em; color: #ffffff;'>{line}</p>" if line.strip() else "<br>")
    
    return '\n'.join(formatted_lines)

def check_api_keys():
    """Check which API keys are configured and return status messages"""
    api_status = []
    missing_keys = []
    
    if FIRECRAWL_API_KEY:
        api_status.append("Firecrawl API ✅")
    else:
        missing_keys.append("Firecrawl API")
    
    if OPENAI_API_KEY:
        api_status.append("OpenAI API ✅")
    else:
        missing_keys.append("OpenAI API")
    
    if ANTHROPIC_API_KEY:
        api_status.append("Anthropic API ✅")
    else:
        missing_keys.append("Anthropic API")
    
    return api_status, missing_keys

def create_combined_pdf(outline, research, blog_content, query):
    """Create a PDF containing all three outputs"""
    class PDF(FPDF):
        def header(self):
            # Add header with date and query
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'R')
            self.cell(0, 10, f'Query: {query}', 0, 1, 'L')
            self.ln(10)

        def footer(self):
            # Add page numbers
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    try:
        pdf = PDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Add SEO Outline
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'SEO Outline', 0, 1, 'C')
        pdf.ln(10)
        pdf.set_font('Arial', '', 12)
        
        # Handle outline content
        for line in outline.split('\n'):
            try:
                # Clean the text to handle special characters
                clean_line = line.encode('latin-1', 'replace').decode('latin-1')
                if line.strip().startswith('#'):
                    # Handle headers
                    level = line.count('#')
                    text = clean_line.strip('#').strip()
                    pdf.set_font('Arial', 'B', 14 - level)
                    pdf.cell(0, 10, text, 0, 1)
                    pdf.set_font('Arial', '', 12)
                else:
                    # Handle normal text
                    pdf.multi_cell(0, 10, clean_line)
            except Exception as e:
                print(f"Error processing line in outline: {e}")
                continue
        
        # Add Research Report
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Research Report', 0, 1, 'C')
        pdf.ln(10)
        pdf.set_font('Arial', '', 12)
        
        # Handle research content
        for line in research.split('\n'):
            try:
                clean_line = line.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 10, clean_line)
            except Exception as e:
                print(f"Error processing line in research: {e}")
                continue
        
        # Add Generated Content
        if blog_content:
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, 'Generated Blog Content', 0, 1, 'C')
            pdf.ln(10)
            pdf.set_font('Arial', '', 12)
            
            # Handle blog content
            for line in blog_content.split('\n'):
                try:
                    clean_line = line.encode('latin-1', 'replace').decode('latin-1')
                    if line.strip().startswith('#'):
                        # Handle headers
                        level = line.count('#')
                        text = clean_line.strip('#').strip()
                        pdf.set_font('Arial', 'B', 14 - level)
                        pdf.cell(0, 10, text, 0, 1)
                        pdf.set_font('Arial', '', 12)
                    else:
                        # Handle normal text
                        pdf.multi_cell(0, 10, clean_line)
                except Exception as e:
                    print(f"Error processing line in blog content: {e}")
                    continue
        
        # Return the PDF as bytes
        return pdf.output(dest='S').encode('latin-1')
    
    except Exception as e:
        raise Exception(f"Error generating PDF: {str(e)}")

def main():
    st.title("SEO Content Analysis Tool")
    st.write("Generate enhanced SEO outlines, research reports, and blog content using AI")

    # Sidebar for API key status
    with st.sidebar:
        st.header("API Status")
        api_status, missing_keys = check_api_keys()
        
        if api_status:
            st.success("\n".join(api_status))
        
        if missing_keys:
            st.error(f"Missing API keys: {', '.join(missing_keys)}")
            st.markdown("""
                Please ensure all required API keys are set in your .env file:
                ```
                FIRECRAWL_API_KEY=your_key_here
                OPENAI_API_KEY=your_key_here
                ANTHROPIC_API_KEY=your_key_here
                ```
            """)
            if not api_status:  # If no APIs are configured
                st.stop()  # Stop execution here

    # User inputs
    search_query = st.text_input("Enter your search query:")
    intent = st.selectbox(
        "Select content intent:",
        ["informational", "commercial", "transactional", "navigational"]
    )
    keywords = st.text_input("Enter secondary keywords (comma-separated):")
    
    # Option to auto-generate content
    auto_generate = st.checkbox("Automatically generate blog content after analysis", value=True)
    
    if st.button("Generate Analysis"):
        if not search_query:
            st.warning("Please enter a search query")
            return
            
        with st.spinner("Processing your request..."):
            try:
                # Create progress container
                progress = st.progress(0)
                status = st.empty()
                
                # Fetch SERP data
                status.text("Fetching SERP data...")
                serp_data = get_search_results(search_query)
                if not serp_data:
                    st.error("Failed to fetch SERP data")
                    return
                progress.progress(20)

                # Perform deep research
                status.text("Performing deep research query...")
                research_output = deep_research(search_query)
                deep_thinking = research_output.get('choices', [{}])[0].get('message', {}).get('content', 'No deep research output provided.')
                citation_urls = research_output.get('citations', [])
                progress.progress(40)

                # Initialize analyzer
                analyzer = LLMEnhancedAnalyzer(
                    firecrawl_api_key=FIRECRAWL_API_KEY,
                    openai_api_key=OPENAI_API_KEY
                )

                # Process keywords
                processed_keywords = [k.strip() for k in keywords.split(',')] if keywords else []
                analyzer.set_content_parameters(intent=intent, keywords=processed_keywords)

                # Extract and scrape competitor URLs
                status.text("Analyzing competitor content...")
                competitor_urls = [result['link'] for result in serp_data.get('organic_results', [])[:5]]
                competitor_data = analyzer.scrape_competitor_content(competitor_urls)
                progress.progress(60)

                # Scrape citation content
                if citation_urls:
                    status.text("Processing citation content...")
                    citation_data = analyzer.scrape_citations(citation_urls)
                else:
                    citation_data = []
                progress.progress(80)

                # Generate outputs
                status.text("Generating final analysis...")
                enhanced_outline = analyzer.generate_enhanced_outline(serp_data, competitor_data, deep_thinking, citation_data)
                research_report = analyzer.generate_research_report(deep_thinking, competitor_data)
                progress.progress(100)

                # Prepare for content generation if auto-generate is enabled
                blog_content = None
                if auto_generate and ANTHROPIC_API_KEY:
                    status.text("Generating blog content... This may take a few minutes.")
                    # Extract research content
                    research_content = extract_research_content(research_report)
                    # Generate blog content
                    blog_content = generate_blog_content(enhanced_outline, research_content)
                    status.empty()
                
                # Clear progress indicators
                status.empty()
                progress.empty()

                # Display results in tabs
                tab1, tab2, tab3 = st.tabs(["SEO Outline", "Research Report", "Generated Content"])
                
                with tab1:
                    st.markdown("### Enhanced SEO Outline")
                    st.markdown(enhanced_outline)
                    if st.download_button(
                        label="Download SEO Outline",
                        data=enhanced_outline,
                        file_name="seo_outline.txt",
                        mime="text/plain"
                    ):
                        st.success("SEO Outline downloaded!")

                with tab2:
                    st.markdown("### Research Report")
                    st.markdown(research_report)
                    if st.download_button(
                        label="Download Research Report",
                        data=research_report,
                        file_name="research_report.txt",
                        mime="text/plain"
                    ):
                        st.success("Research Report downloaded!")
                
                with tab3:
                    st.markdown("### Generated Blog Content")
                    
                    # If content was auto-generated, display it
                    if blog_content:
                        # Format and display content
                        formatted_content = format_blog_content(blog_content)
                        st.markdown(formatted_content, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.download_button(
                                label="Download as Text",
                                data=blog_content,
                                file_name="generated_blog.txt",
                                mime="text/plain"
                            ):
                                st.success("Blog content downloaded!")
                        
                        with col2:
                            if st.download_button(
                                label="Download as Markdown",
                                data=blog_content,
                                file_name="generated_blog.md",
                                mime="text/markdown"
                            ):
                                st.success("Blog content downloaded as Markdown!")
                    # If auto-generate is disabled, show the generate button
                    elif not auto_generate:
                        if st.button("Generate Blog Content"):
                            with st.spinner("Generating blog content... This may take a few minutes."):
                                try:
                                    # Extract research content
                                    research_content = extract_research_content(research_report)
                                    
                                    # Generate blog content
                                    blog_content = generate_blog_content(enhanced_outline, research_content)
                                    
                                    if blog_content:
                                        # Format and display content
                                        formatted_content = format_blog_content(blog_content)
                                        st.markdown(formatted_content, unsafe_allow_html=True)
                                        
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            if st.download_button(
                                                label="Download as Text",
                                                data=blog_content,
                                                file_name="generated_blog.txt",
                                                mime="text/plain"
                                            ):
                                                st.success("Blog content downloaded!")
                                        
                                        with col2:
                                            if st.download_button(
                                                label="Download as Markdown",
                                                data=blog_content,
                                                file_name="generated_blog.md",
                                                mime="text/markdown"
                                            ):
                                                st.success("Blog content downloaded as Markdown!")
                                    else:
                                        st.error("No content was generated. Please check your API key and try again.")
                                except Exception as e:
                                    st.error(f"Error during content generation: {str(e)}")
                                    st.info("Please make sure your ANTHROPIC_API_KEY is correctly set in the .env file")
                    else:
                        st.error("Content generation failed or was not attempted. Please check your API keys.")

                # Add a section for combined PDF download
                st.markdown("### Download Combined Report")
                if st.button("Generate Combined PDF"):
                    try:
                        with st.spinner("Generating PDF..."):
                            pdf_bytes = create_combined_pdf(
                                enhanced_outline,
                                research_report,
                                blog_content if 'blog_content' in locals() else None,
                                search_query
                            )
                            
                            st.download_button(
                                label="Download Complete Report as PDF",
                                data=pdf_bytes,
                                file_name=f"seo_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                mime="application/pdf"
                            )
                            st.success("PDF generated successfully!")
                    except Exception as e:
                        st.error(f"Error generating PDF: {str(e)}")
                        st.info("If you're seeing encoding errors, this might be due to special characters in the content.")

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()