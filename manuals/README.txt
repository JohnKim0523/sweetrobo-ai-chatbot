MANUAL ORGANIZATION GUIDE
=========================

Add your PDF manuals to this folder structure:

MACHINE-SPECIFIC MANUALS:
-------------------------
manuals/cotton_candy/
  - user_manual.pdf
  - error_codes.pdf
  - maintenance_guide.pdf

manuals/ice_cream/
  - user_manual.pdf
  - troubleshooting.pdf

manuals/balloon_bot/
  - user_manual.pdf

manuals/candy_monster/
  - user_manual.pdf

manuals/popcart/
  - user_manual.pdf

manuals/mr_pop/
  - user_manual.pdf

manuals/marshmallow_spaceship/
  - user_manual.pdf


GENERAL MANUALS (All machines):
--------------------------------
manuals/
  - common_issues.pdf
  - safety_guidelines.pdf
  - wifi_setup.pdf
  - payment_system_guide.pdf


PROCESSING:
-----------
After adding PDFs, run:
  python process_manuals.py

This will:
1. Extract text from all PDFs
2. Split into searchable chunks
3. Create embeddings
4. Upload to Pinecone

The chatbot will then be able to answer questions using
information from both the Q&A dataset AND the manuals!
