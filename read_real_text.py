import pandas as pd

def read_text_data():
    print("How2Sign ki asal Text file padhi ja rahi hai...")
    file_path = "how2sign_train.csv"
    
    try:
        # Pehle Tab (\t) se try karein
        df = pd.read_csv(file_path, sep='\t', on_bad_lines='skip')
        
        # Agar sirf 1 hi column bana ho, toh comma (,) se try karein
        if len(df.columns) == 1:
            df = pd.read_csv(file_path, sep=',', on_bad_lines='skip')
            
        print(f"Total Rows (Sentences) mile: {len(df)}")
        print("\nPehle 5 sentences (Asal Data):")
        
        # Text wala column dhoondein (aksar SENTENCE ya last column hota hai)
        text_col = 'SENTENCE' if 'SENTENCE' in df.columns else df.columns[-1]
        
        for i in range(min(5, len(df))):
            print(f"- {df.iloc[i][text_col]}")
            
        print("\nReal Text Data successfully load ho gaya!")
        
    except Exception as e:
        print("File padhne mein error aya:", e)

if __name__ == "__main__":
    read_text_data()