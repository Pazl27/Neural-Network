package org.example.utils;

import javax.swing.*;
import java.io.OutputStream;

public class CustomOutputStream extends OutputStream {

    private final JTextArea textArea;

    public CustomOutputStream(JTextArea textArea) {
        this.textArea = textArea;
    }

    @Override
    public void write(int b) {
        textArea.append(String.valueOf((char) b));
        textArea.setCaretPosition(textArea.getDocument().getLength());
    }
}
